import csv
import os
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import gymnasium as gym
except ImportError:
    import gym

from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer, TerrainGymEnv


class FailureRecordingWrapper(gym.Wrapper):
    """Record failure episodes as transition chains and periodically dump to PKL."""

    def __init__(self, env, out_dir: str, pkl_name: str = "train_failure_chains.pkl", flush_every_episodes: int = 50):
        super().__init__(env)
        self.out_dir = out_dir
        self.pkl_path = os.path.join(out_dir, pkl_name)
        self.flush_every_episodes = int(max(1, flush_every_episodes))

        self._curr_obs = None
        self._curr_chain = []
        self._episode_idx = 0
        self._failure_episodes = []

        os.makedirs(self.out_dir, exist_ok=True)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._curr_obs = np.asarray(obs, dtype=np.float32)
        self._curr_chain = []
        return obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        transition = {
            "obs": np.asarray(self._curr_obs, dtype=np.float32).tolist() if self._curr_obs is not None else [],
            "action": np.asarray(action, dtype=np.float32).tolist(),
            "reward": float(reward),
            "next_obs": np.asarray(next_obs, dtype=np.float32).tolist(),
            "done": done,
            "terrain_control": np.asarray(action, dtype=np.float32).tolist(),
            "info": info,
        }
        self._curr_chain.append(transition)
        self._curr_obs = np.asarray(next_obs, dtype=np.float32)

        if done:
            has_failure = bool(
                info.get("fallen", False)
                or info.get("base_collision", False)
                or info.get("thigh_collision", False)
                or info.get("stuck", False)
            )
            if has_failure:
                self._failure_episodes.append({"episode": int(self._episode_idx), "chain": self._curr_chain})

            self._episode_idx += 1
            if self._episode_idx % self.flush_every_episodes == 0:
                self._flush()

        return next_obs, reward, terminated, truncated, info

    def _flush(self):
        with open(self.pkl_path, "wb") as f:
            pickle.dump(self._failure_episodes, f)

    def close(self):
        self._flush()
        return self.env.close()


class TrainingLoggerCallback(BaseCallback):
    """Unified logger for PPO/SAC so plotting remains consistent."""

    def __init__(self, out_dir: str, save_every_steps: int = 2000, smooth_window: int = 20, checkpoint_every_steps: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.out_dir = out_dir
        self.save_every_steps = int(max(1, save_every_steps))
        self.smooth_window = int(max(1, smooth_window))
        self.checkpoint_every_steps = int(max(1, checkpoint_every_steps))

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_timesteps = []
        self.actor_losses = []
        self.critic_losses = []
        self.loss_timesteps = []

        os.makedirs(self.out_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                ep = info["episode"]
                self.episode_rewards.append(float(ep.get("r", 0.0)))
                self.episode_lengths.append(float(ep.get("l", 0.0)))
                self.episode_timesteps.append(int(self.num_timesteps))

        if hasattr(self.model, "logger"):
            log_dict = self.model.logger.name_to_value
            actor = None
            critic = None
            for k in ["train/actor_loss", "train/policy_gradient_loss"]:
                if k in log_dict:
                    actor = float(log_dict[k])
                    break
            for k in ["train/critic_loss", "train/value_loss"]:
                if k in log_dict:
                    critic = float(log_dict[k])
                    break
            if actor is not None:
                self.actor_losses.append(actor)
            if critic is not None:
                self.critic_losses.append(critic)
            if actor is not None or critic is not None:
                self.loss_timesteps.append(int(self.num_timesteps))

        if self.num_timesteps % self.save_every_steps == 0:
            self._dump_metrics_csv("metrics_partial.csv")
            self._save_combined_plot("training_curves_partial.png")

        if self.num_timesteps % self.checkpoint_every_steps == 0:
            ckpt_path = os.path.join(self.out_dir, f"checkpoint_step_{self.num_timesteps}.zip")
            self.model.save(ckpt_path)

        return True

    def _on_training_end(self) -> None:
        self._dump_metrics_csv("metrics.csv")
        self._save_combined_plot("training_curves.png")

    def _smooth(self, values):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size < self.smooth_window:
            return arr
        kernel = np.ones((self.smooth_window,), dtype=np.float32) / float(self.smooth_window)
        return np.convolve(arr, kernel, mode="valid")

    def _dump_metrics_csv(self, filename: str):
        csv_path = os.path.join(self.out_dir, filename)
        max_len = max(len(self.episode_rewards), len(self.actor_losses), len(self.critic_losses))
        if max_len == 0:
            return

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode_timestep",
                "episode_reward",
                "episode_length",
                "loss_timestep",
                "actor_loss",
                "critic_loss",
            ])
            for i in range(max_len):
                writer.writerow([
                    self.episode_timesteps[i] if i < len(self.episode_timesteps) else "",
                    self.episode_rewards[i] if i < len(self.episode_rewards) else "",
                    self.episode_lengths[i] if i < len(self.episode_lengths) else "",
                    self.loss_timesteps[i] if i < len(self.loss_timesteps) else "",
                    self.actor_losses[i] if i < len(self.actor_losses) else "",
                    self.critic_losses[i] if i < len(self.critic_losses) else "",
                ])

    def _save_combined_plot(self, filename: str):
        if len(self.episode_rewards) == 0 and len(self.actor_losses) == 0 and len(self.critic_losses) == 0:
            return

        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        if len(self.episode_rewards) > 0:
            axes[0].plot(self.episode_timesteps, self.episode_rewards, linewidth=1.0, label="reward")
            smoothed = self._smooth(self.episode_rewards)
            if smoothed.size > 0 and smoothed.size != len(self.episode_rewards):
                offset = len(self.episode_rewards) - smoothed.size
                axes[0].plot(self.episode_timesteps[offset:], smoothed, linewidth=2.0, label=f"reward_ma{self.smooth_window}")
            axes[0].set_title("Episode Reward")
            axes[0].grid(True)
            axes[0].legend(loc="best")

        if len(self.actor_losses) > 0:
            x_actor = self.loss_timesteps[:len(self.actor_losses)] if len(self.loss_timesteps) >= len(self.actor_losses) else np.arange(len(self.actor_losses))
            axes[1].plot(x_actor, self.actor_losses, linewidth=1.0, label="actor_loss")
            axes[1].set_title("Actor Loss")
            axes[1].grid(True)
            axes[1].legend(loc="best")

        if len(self.critic_losses) > 0:
            x_critic = self.loss_timesteps[:len(self.critic_losses)] if len(self.loss_timesteps) >= len(self.critic_losses) else np.arange(len(self.critic_losses))
            axes[2].plot(x_critic, self.critic_losses, linewidth=1.0, label="critic_loss")
            axes[2].set_title("Critic Loss")
            axes[2].grid(True)
            axes[2].legend(loc="best")

        for ax in axes:
            ax.set_xlabel("Timesteps")

        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, filename), dpi=150)
        plt.close(fig)


def configure_torch_runtime(cfg: dict):
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


def train_ppo(go2_cfg, terrain_cfg, total_timesteps=20000, max_episode_steps=35, log_dir="train_terrain_logs", device="auto", learning_rate=3e-4, batch_size=256, n_steps=2048, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, seed=0, plot_save_every_steps=2000, plot_smooth_window=20, checkpoint_every_steps=10000, preload_model_path="", failure_pkl_name="train_failure_chains.pkl", failure_flush_every_episodes=50):

    def make_env():
        trainer = TerrainTrainer(go2_cfg, terrain_cfg)
        env = TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        env = FailureRecordingWrapper(
            env,
            out_dir=log_dir,
            pkl_name=failure_pkl_name,
            flush_every_episodes=failure_flush_every_episodes,
        )
        return env

    vec_env = DummyVecEnv([make_env])

    action_dim = int(np.prod(vec_env.action_space.shape))
    if action_dim <= 0:
        raise ValueError(
            "Terrain action dimension is 0, PPO cannot train with empty action space. "
            "Please check terrain config and ensure terrain_action.terrain_types includes at least one controllable type "
            "(e.g. ['bump'])."
        )

    if preload_model_path and os.path.exists(preload_model_path):
        print(f"[train_PPO] loading pretrained model from: {preload_model_path}")
        model = PPO.load(preload_model_path, env=vec_env, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device=device,
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            n_steps=int(n_steps),
            n_epochs=int(n_epochs),
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
            clip_range=float(clip_range),
            ent_coef=float(ent_coef),
            vf_coef=float(vf_coef),
            max_grad_norm=float(max_grad_norm),
            seed=int(seed),
        )

    callback = TrainingLoggerCallback(
        out_dir=log_dir,
        save_every_steps=int(plot_save_every_steps),
        smooth_window=int(plot_smooth_window),
        checkpoint_every_steps=int(checkpoint_every_steps),
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(log_dir, "model.zip"))


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    train_config_file = "train_config.yaml"
    with open(os.path.join(current_path, train_config_file), "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)

    configure_torch_runtime(train_config)

    log_dir = os.path.join(current_path, "train_logs", train_config["log_name"])
    os.makedirs(log_dir, exist_ok=True)

    terrain_cfg_file = os.path.join(current_path, train_config["terrain_config"])
    train_cfg_file = os.path.join(current_path, train_config_file)
    if os.path.exists(terrain_cfg_file):
        shutil.copy2(terrain_cfg_file, log_dir)
    shutil.copy2(train_cfg_file, log_dir)

    go2_cfg = [train_config["go2_task"], train_config["go2_config"]]
    terrain_cfg = f"train_PPO/{train_config['terrain_config']}"

    train_ppo(
        go2_cfg=go2_cfg,
        terrain_cfg=terrain_cfg,
        total_timesteps=train_config.get("total_timesteps", 20000),
        max_episode_steps=train_config.get("max_episode_steps", 35),
        log_dir=log_dir,
        device=train_config.get("device", "auto"),
        learning_rate=train_config.get("learning_rate", 3e-4),
        batch_size=train_config.get("batch_size", 256),
        n_steps=train_config.get("n_steps", 2048),
        n_epochs=train_config.get("n_epochs", 10),
        gamma=train_config.get("gamma", 0.99),
        gae_lambda=train_config.get("gae_lambda", 0.95),
        clip_range=train_config.get("clip_range", 0.2),
        ent_coef=train_config.get("ent_coef", 0.0),
        vf_coef=train_config.get("vf_coef", 0.5),
        max_grad_norm=train_config.get("max_grad_norm", 0.5),
        seed=train_config.get("seed", 0),
        plot_save_every_steps=train_config.get("plot_save_every_steps", 2000),
        plot_smooth_window=train_config.get("plot_smooth_window", 20),
        checkpoint_every_steps=train_config.get("checkpoint_every_steps", 10000),
        preload_model_path=train_config.get("preload_model_path", ""),
        failure_pkl_name=train_config.get("failure_pkl_name", "train_failure_chains.pkl"),
        failure_flush_every_episodes=train_config.get("failure_flush_every_episodes", 50),
    )


if __name__ == "__main__":
    main()

