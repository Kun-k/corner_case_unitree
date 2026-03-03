import os
import time
import mujoco
import numpy as np

from terrain_trainer import TerrainTrainer, TerrainGymEnv

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    SAC = None


def train_sac(config_path: str, model_path: str = "sac_terrain", total_timesteps: int = 10000):
    trainer = TerrainTrainer(config_path, terrain_types=['bump'], terrain_decimation=20, render=True)
    env = TerrainGymEnv(trainer, steps_per_terrain=20, max_episode_steps=200)

    if SAC is None:
        raise RuntimeError("stable-baselines3 not available. Install with: pip install stable-baselines3")

    # SB3 requires a vectorized env; we will wrap and then train inside the viewer
    vec_env = DummyVecEnv([lambda: env])

    model = SAC('MlpPolicy', vec_env, verbose=1)

    # Run training inside the viewer to keep mujoco viewer active
    with mujoco.viewer.launch_passive(trainer.model, trainer.data) as viewer:
        # Reset trainer to ensure viewer has control
        trainer.reset()
        # Train for a number of environment steps (SB3 steps = environment steps)
        model.learn(total_timesteps=total_timesteps)

    # save model
    model.save(model_path)


if __name__ == "__main__":
    cfg = os.path.join("velocity", "configs", "go2.yaml")
    train_sac(cfg, model_path="sac_terrain.zip", total_timesteps=20000)
