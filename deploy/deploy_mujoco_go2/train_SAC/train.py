import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer, TerrainGymEnv


class TrainingLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []

    def _on_step(self) -> bool:

        # 记录 episode reward
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        # 记录 loss
        if hasattr(self.model, "logger"):
            log_dict = self.model.logger.name_to_value

            if "train/actor_loss" in log_dict:
                self.actor_losses.append(log_dict["train/actor_loss"])

            if "train/critic_loss" in log_dict:
                self.critic_losses.append(log_dict["train/critic_loss"])

        return True


def plot_training(callback, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(callback.episode_rewards)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig(os.path.join(out_dir, "episode_reward.png"), dpi=140)
    plt.close()

    if len(callback.actor_losses) > 0:
        plt.figure()
        plt.plot(callback.actor_losses)
        plt.title("Actor Loss")
        plt.grid()
        plt.savefig(os.path.join(out_dir, "actor_loss.png"), dpi=140)
        plt.close()

    if len(callback.critic_losses) > 0:
        plt.figure()
        plt.plot(callback.critic_losses)
        plt.title("Critic Loss")
        plt.grid()
        plt.savefig(os.path.join(out_dir, "critic_loss.png"), dpi=140)
        plt.close()


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


def train_sac(go2_cfg, terrain_cfg,
              total_timesteps=20000,
              max_episode_steps=35,
              log_dir="train_terrain_logs",
              device="auto",
              learning_rate=3e-4,
              batch_size=256,
              buffer_size=1_000_000,
              learning_starts=100,
              train_freq=1,
              gradient_steps=1,
              tau=0.005,
              gamma=0.99,
              seed=0):

    def make_env():
        trainer = TerrainTrainer(go2_cfg, terrain_cfg)
        env = TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_env])

    action_dim = int(np.prod(vec_env.action_space.shape))
    if action_dim <= 0:
        raise ValueError(
            "Terrain action dimension is 0, SAC cannot train with empty action space. "
            "Please check terrain config and ensure terrain_action.terrain_types includes at least one controllable type "
            "(e.g. ['bump'])."
        )

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        buffer_size=int(buffer_size),
        learning_starts=int(learning_starts),
        train_freq=int(train_freq),
        gradient_steps=int(gradient_steps),
        tau=float(tau),
        gamma=float(gamma),
        seed=int(seed),
    )

    callback = TrainingLoggerCallback()

    model.learn(total_timesteps=total_timesteps,
                callback=callback)

    model.save(os.path.join(log_dir, "model.zip"))

    plot_training(callback, log_dir)


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

    go2_cfg = [train_config['go2_task'], train_config['go2_config']]
    terrain_cfg = f"train_SAC/{train_config['terrain_config']}"
    total_timesteps = train_config['total_timesteps']
    max_episode_steps = train_config['max_episode_steps']

    train_sac(
        go2_cfg,
        terrain_cfg,
        total_timesteps=total_timesteps,
        max_episode_steps=max_episode_steps,
        log_dir=log_dir,
        device=train_config.get("device", "auto"),
        learning_rate=train_config.get("learning_rate", 3e-4),
        batch_size=train_config.get("batch_size", 256),
        buffer_size=train_config.get("buffer_size", 1_000_000),
        learning_starts=train_config.get("learning_starts", 100),
        train_freq=train_config.get("train_freq", 1),
        gradient_steps=train_config.get("gradient_steps", 1),
        tau=train_config.get("tau", 0.005),
        gamma=train_config.get("gamma", 0.99),
        seed=train_config.get("seed", 0),
    )


if __name__ == "__main__":
    main()
