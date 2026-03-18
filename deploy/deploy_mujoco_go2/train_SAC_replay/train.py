import os
import random
from typing import Dict, List, Optional

import mujoco
import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import gymnasium as gym
    from gymnasium.spaces import Box
except ImportError:
    import gym
    from gym.spaces import Box

from deploy.deploy_mujoco_go2.classifier_gate import ClassifierGate
from deploy.deploy_mujoco_go2.offline_data_utils import collect_pkl_files, load_chains_from_pkl_file
from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer
from deploy.deploy_mujoco_go2.train_SAC.train import (
    FilteredGatedReplayBuffer,
    GatedSAC,
    TrainingLoggerCallback,
    _extract_reward_cfg_from_terrain_yaml,
    configure_torch_runtime,
    preload_replay_buffer_from_pkl,
)


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_state_from_trace(trainer: TerrainTrainer, tr: Dict) -> bool:
    info = tr.get("info", {}) if isinstance(tr.get("info"), dict) else {}
    trace = info.get("go2_rollout_trace", {}) if isinstance(info.get("go2_rollout_trace"), dict) else {}
    states = trace.get("states", [])
    if len(states) == 0:
        return False

    st0 = states[0]
    qpos = np.asarray(st0.get("qpos", []), dtype=np.float64)
    qvel = np.asarray(st0.get("qvel", []), dtype=np.float64)
    if qpos.shape[0] != trainer.data.qpos.shape[0] or qvel.shape[0] != trainer.data.qvel.shape[0]:
        return False

    trainer.data.qpos[:] = qpos
    trainer.data.qvel[:] = qvel
    mujoco.mj_forward(trainer.model, trainer.data)
    return True


def _get_obs_from_transition(trainer: TerrainTrainer, tr: Dict) -> np.ndarray:
    # Prefer replayed obs for strict frame-by-frame offline chain exposure.
    obs = np.asarray(tr.get("obs", []), dtype=np.float32)
    if obs.ndim == 1 and obs.shape[0] == trainer.get_terrain_observation().shape[0]:
        return obs
    # Fallback to simulator-derived obs.
    return trainer.get_terrain_observation().astype(np.float32)


def _load_replay_chains(paths: List[str], keep_k: int) -> List[List[Dict]]:
    files = collect_pkl_files(paths)
    chains_all: List[List[Dict]] = []
    for fp in files:
        try:
            chains = load_chains_from_pkl_file(fp, consecutive_fail_keep_k=int(keep_k))
            chains_all.extend(chains)
        except Exception:
            continue
    return chains_all


class ReplayTerrainGymEnv(gym.Env):
    """Replay-chain driven terrain env for SAC.

    At each RL step:
    1) move simulator to current replay frame state (if trace exists)
    2) SAC outputs terrain action
    3) execute action with trainer.step(...)
    4) store new transition in SAC buffer
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        trainer: TerrainTrainer,
        replay_chains: List[List[Dict]],
        max_episode_steps: int,
        require_trace: bool = False,
        random_chain: bool = True,
    ):
        super().__init__()
        self.trainer = trainer
        self.replay_chains = replay_chains
        self.max_episode_steps = int(max_episode_steps)
        self.require_trace = bool(require_trace)
        self.random_chain = bool(random_chain)

        obs_dim = int(self.trainer.get_terrain_observation().shape[0])
        act_dim = int(self.trainer.total_action_dims)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

        self._chain_cursor = -1
        self._curr_chain: Optional[List[Dict]] = None
        self._step_idx = 0

    def _pick_chain(self) -> List[Dict]:
        if len(self.replay_chains) == 0:
            raise RuntimeError("No replay chains loaded")
        if self.random_chain:
            return random.choice(self.replay_chains)
        self._chain_cursor = (self._chain_cursor + 1) % len(self.replay_chains)
        return self.replay_chains[self._chain_cursor]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.trainer.reset()
        self._curr_chain = self._pick_chain()
        self._step_idx = 0

        if len(self._curr_chain) == 0:
            obs = self.trainer.get_terrain_observation().astype(np.float32)
            return obs, {}

        first_tr = self._curr_chain[0]
        ok = _set_state_from_trace(self.trainer, first_tr)
        if self.require_trace and not ok:
            raise RuntimeError("replay_require_trace=true but trace state missing or invalid")

        obs = _get_obs_from_transition(self.trainer, first_tr)
        return obs, {}

    def step(self, action):
        if self._curr_chain is None or len(self._curr_chain) == 0:
            raise RuntimeError("Current replay chain is empty")

        tr = self._curr_chain[min(self._step_idx, len(self._curr_chain) - 1)]
        ok = _set_state_from_trace(self.trainer, tr)
        if self.require_trace and not ok:
            raise RuntimeError("replay_require_trace=true but trace state missing or invalid")

        next_obs, _, reward, done_sim, info = self.trainer.step(np.asarray(action, dtype=np.float32))
        self._step_idx += 1

        truncated = self._step_idx >= min(self.max_episode_steps, len(self._curr_chain))
        terminated = bool(done_sim)

        # Helpful flags in infos for downstream filtering/debug.
        info = dict(info) if isinstance(info, dict) else {}
        info.setdefault("from_replay_chain", True)
        info.setdefault("replay_step_idx", int(self._step_idx))

        return next_obs.astype(np.float32), float(reward), terminated, truncated, info


def train_sac_replay(go2_cfg, terrain_cfg, cfg: Dict, reward_cfg: Dict, log_dir: str) -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    replay_pkl_paths = []
    preload_pkl_paths = []
    for path in cfg.get("replay_pkl_paths", []):
        replay_pkl_paths.append(os.path.join(current_dir, path))
    for path in cfg.get("preload_pkl_paths", []):
        preload_pkl_paths.append(os.path.join(current_dir, path))

    replay_chains = _load_replay_chains(
        paths=replay_pkl_paths,
        keep_k=int(cfg.get("consecutive_fail_keep_k", 0)),
    )
    if len(replay_chains) == 0:
        raise RuntimeError("No replay chains loaded from train_config.yaml:replay_pkl_paths")

    def make_env():
        trainer = TerrainTrainer(go2_cfg, terrain_cfg)
        env = ReplayTerrainGymEnv(
            trainer=trainer,
            replay_chains=replay_chains,
            max_episode_steps=int(cfg.get("max_episode_steps", 35)),
            require_trace=bool(cfg.get("replay_require_trace", False)),
            random_chain=bool(cfg.get("replay_random_chain", True)),
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    classifier_gate = None
    gate_cfg = cfg.get("classifier_gate", {})
    if bool(gate_cfg.get("enabled", False)):
        ckpt = str(gate_cfg.get("checkpoint_path", "")).strip()
        if not ckpt:
            raise ValueError("classifier_gate.enabled=true but checkpoint_path is empty")
        if not os.path.isabs(ckpt):
            ckpt = os.path.normpath(os.path.join(os.path.dirname(__file__), ckpt))
        classifier_gate = ClassifierGate(
            checkpoint_path=ckpt,
            device=str(gate_cfg.get("device", cfg.get("device", "cpu"))),
            hidden_dim=int(gate_cfg.get("hidden_dim", 1024)),
            concat_action_to_obs=bool(gate_cfg.get("concat_action_to_obs", True)),
        )

    gated_mode = classifier_gate is not None
    model_cls = GatedSAC if gated_mode else SAC
    model_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        device=cfg.get("device", "auto"),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        batch_size=int(cfg.get("batch_size", 256)),
        buffer_size=int(cfg.get("buffer_size", 1_000_000)),
        learning_starts=int(cfg.get("learning_starts", 100)),
        train_freq=int(cfg.get("train_freq", 1)),
        gradient_steps=int(cfg.get("gradient_steps", 1)),
        tau=float(cfg.get("tau", 0.005)),
        gamma=float(cfg.get("gamma", 0.99)),
        seed=int(cfg.get("seed", 0)),
        replay_buffer_class=FilteredGatedReplayBuffer,
        replay_buffer_kwargs={"consecutive_fail_keep_k": int(cfg.get("consecutive_fail_keep_k", 0))},
    )
    if gated_mode:
        model_kwargs.update(
            gate=classifier_gate,
            gate_threshold=float(gate_cfg.get("threshold", 0.5)),
            min_rl_buffer_size_for_update=int(gate_cfg.get("min_rl_buffer_size_for_update", cfg.get("learning_starts", 100))),
            consecutive_fail_keep_k=int(cfg.get("consecutive_fail_keep_k", 0)),
        )

    model = model_cls(**model_kwargs)

    preload_model = str(cfg.get("preload_model_path", "")).strip()
    if preload_model:
        if not os.path.isabs(preload_model):
            preload_model = os.path.normpath(os.path.join(os.path.dirname(__file__), preload_model))
        if os.path.exists(preload_model):
            model.set_parameters(preload_model, exact_match=False, device=cfg.get("device", "auto"))

    # Task requirement: offline pkl data should also enter replay buffer.
    preload_paths = preload_pkl_paths + replay_pkl_paths
    if len(preload_paths) > 0:
        base = os.path.dirname(os.path.realpath(__file__))
        resolved = [p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p)) for p in preload_paths]
        preload_replay_buffer_from_pkl(
            model,
            resolved,
            reward_cfg=reward_cfg,
            consecutive_fail_keep_k=int(cfg.get("consecutive_fail_keep_k", 0)),
        )

    callback = TrainingLoggerCallback(
        out_dir=log_dir,
        save_every_steps=int(cfg.get("plot_save_every_steps", 2000)),
        smooth_window=int(cfg.get("plot_smooth_window", 20)),
        checkpoint_every_steps=int(cfg.get("checkpoint_every_steps", 10000)),
    )

    model.learn(total_timesteps=int(cfg.get("total_timesteps", 80_000)), callback=callback)
    model.save(os.path.join(log_dir, "model.zip"))


def main() -> None:
    current_path = os.path.dirname(os.path.realpath(__file__))
    train_cfg_path = os.path.join(current_path, "train_config.yaml")
    cfg = _load_yaml(train_cfg_path)

    configure_torch_runtime(cfg)

    log_dir = os.path.join(current_path, "train_logs", str(cfg.get("log_name", "sac_replay_default")))
    os.makedirs(log_dir, exist_ok=True)

    terrain_cfg_path = os.path.join(current_path, str(cfg.get("terrain_config", "terrain_config.yaml")))
    reward_cfg = _extract_reward_cfg_from_terrain_yaml(terrain_cfg_path)

    # snapshot configs
    shutil_ok = True
    try:
        import shutil

        shutil.copy2(train_cfg_path, log_dir)
        if os.path.exists(terrain_cfg_path):
            shutil.copy2(terrain_cfg_path, log_dir)
    except Exception:
        shutil_ok = False

    if not shutil_ok:
        print("[train_SAC_replay] warning: failed to copy config snapshots into log_dir")

    go2_cfg = [cfg.get("go2_task", "terrain"), cfg.get("go2_config", "go2.yaml")]
    terrain_cfg_rel = f"train_SAC_replay/{cfg.get('terrain_config', 'terrain_config.yaml')}"

    train_sac_replay(
        go2_cfg=go2_cfg,
        terrain_cfg=terrain_cfg_rel,
        cfg=cfg,
        reward_cfg=reward_cfg,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()

