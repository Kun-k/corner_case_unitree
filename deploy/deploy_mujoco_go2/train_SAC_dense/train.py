import os
import shutil

import numpy as np
import torch
import yaml
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan

from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer, TerrainGymEnv
from deploy.deploy_mujoco_go2.train_SAC_dense.callbacks import DenseTrainingLogger
from deploy.deploy_mujoco_go2.train_SAC_dense.dense_replay_buffer import FailureReplayBuffer


class FiniteValueWrapper(gym.Wrapper):
    """Clamp non-finite obs/reward/action to keep SAC numerically stable."""

    def __init__(self, env: gym.Env, obs_abs_clip: float = 1e3):
        super().__init__(env)
        self.obs_abs_clip = float(obs_abs_clip)

    def _sanitize_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=self.obs_abs_clip, neginf=-self.obs_abs_clip)
        return np.clip(obs, -self.obs_abs_clip, self.obs_abs_clip)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._sanitize_obs(obs), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.size > 0:
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._sanitize_obs(obs)
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        return obs, reward, terminated, truncated, info


def configure_torch_runtime(cfg: dict):
    """Apply torch/cuda runtime knobs from train_config."""
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    deterministic = bool(cfg.get("torch_deterministic", False))
    cudnn_benchmark = bool(cfg.get("cudnn_benchmark", True))
    allow_tf32 = bool(cfg.get("allow_tf32", True))

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    num_threads = int(cfg.get("torch_num_threads", 0))
    if num_threads > 0:
        torch.set_num_threads(num_threads)


def train_sac_dense(
    go2_cfg,
    terrain_cfg,
    total_timesteps=50000,
    max_episode_steps=350,
    model_path="sac_dense_model.zip",
    log_dir="train_terrain_logs_dense",
    reward_threshold=8.0,
    learning_starts=10000,
    device="auto",
    learning_rate=1e-4,
    batch_size=256,
    buffer_size=1_000_000,
    train_freq=1,
    gradient_steps=1,
    tau=0.005,
    gamma=0.99,
    seed=0,
    obs_abs_clip=1e3,
    plot_save_every_steps=2000,
    plot_smooth_window=20,
):
    os.makedirs(log_dir, exist_ok=True)

    def make_env():
        trainer = TerrainTrainer(go2_cfg, terrain_cfg)
        env = TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)
        env = FiniteValueWrapper(env, obs_abs_clip=obs_abs_clip)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecCheckNan(vec_env, raise_exception=False, check_inf=True)

    action_dim = int(np.prod(vec_env.action_space.shape))
    if action_dim <= 0:
        raise ValueError(
            "Terrain action dimension is 0, SAC dense cannot train with empty action space. "
            "Please check terrain config and ensure terrain_action.terrain_types includes controllable types (e.g. ['bump'])."
        )

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_starts=int(learning_starts),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        buffer_size=int(buffer_size),
        train_freq=int(train_freq),
        gradient_steps=int(gradient_steps),
        tau=float(tau),
        gamma=float(gamma),
        replay_buffer_class=FailureReplayBuffer,
        replay_buffer_kwargs={"reward_threshold": float(reward_threshold)},
        device=device,
        seed=int(seed),
    )

    callback = DenseTrainingLogger(
        out_dir=log_dir,
        save_every_steps=int(plot_save_every_steps),
        smooth_window=int(plot_smooth_window),
    )
    model.learn(total_timesteps=int(total_timesteps), callback=callback)
    model.save(model_path)
    return model_path


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    train_config_file = "train_config.yaml"
    with open(f"{current_path}/{train_config_file}", "r", encoding="utf-8") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    configure_torch_runtime(train_config)

    log_dir = f"{current_path}/train_logs/{train_config['log_name']}"
    os.makedirs(log_dir, exist_ok=True)

    terrain_cfg_file = os.path.join(current_path, train_config["terrain_config"])
    train_cfg_file = os.path.join(current_path, train_config_file)
    os.system(f"cp {terrain_cfg_file} {log_dir}")
    os.system(f"cp {train_cfg_file} {log_dir}")

    go2_cfg = [train_config["go2_task"], train_config["go2_config"]]
    terrain_cfg = f"train_SAC_dense/{train_config['terrain_config']}"

    train_sac_dense(
        go2_cfg,
        terrain_cfg,
        total_timesteps=train_config["total_timesteps"],
        max_episode_steps=train_config["max_episode_steps"],
        model_path=os.path.join(log_dir, "model.zip"),
        log_dir=log_dir,
        reward_threshold=train_config.get("reward_threshold", 8.0),
        learning_starts=train_config.get("learning_starts", 10000),
        device=train_config.get("device", "auto"),
        learning_rate=train_config.get("learning_rate", 1e-4),
        batch_size=train_config.get("batch_size", 256),
        buffer_size=train_config.get("buffer_size", 1_000_000),
        train_freq=train_config.get("train_freq", 1),
        gradient_steps=train_config.get("gradient_steps", 1),
        tau=train_config.get("tau", 0.005),
        gamma=train_config.get("gamma", 0.99),
        seed=train_config.get("seed", 0),
        obs_abs_clip=train_config.get("obs_abs_clip", 1e3),
        plot_save_every_steps=train_config.get("plot_save_every_steps", 2000),
        plot_smooth_window=train_config.get("plot_smooth_window", 20),
    )


if __name__ == "__main__":
    main()
