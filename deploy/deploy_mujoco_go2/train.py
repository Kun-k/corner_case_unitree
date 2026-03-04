from terrain_trainer import TerrainTrainer, TerrainGymEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv


def train_sac(go2_cfg, terrain_cfg, total_timesteps=20000, max_episode_steps=35, model_path="sac_model.zip"):
    trainer = TerrainTrainer(go2_cfg, terrain_cfg)
    env = TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)
    vec_env = DummyVecEnv([lambda: env])

    model = SAC('MlpPolicy', vec_env, verbose=1)

    # Run training inside the viewer to keep mujoco viewer active
    # Reset trainer to ensure viewer has control
    # trainer.reset()
    # Train for a number of environment steps (SB3 steps = environment steps)
    model.learn(total_timesteps=total_timesteps)
    trainer.close_viewer()

    # save model
    model.save(model_path)


if __name__ == "__main__":
    # TODO 参数化
    # TODO 绘图
    go2_cfg = ["terrain", "go2.yaml"]
    terrain_cfg = "terrain_config.yaml"
    total_timesteps = 20000
    max_episode_steps = 35
    model_path = "sac_model.zip"
    train_sac(go2_cfg, terrain_cfg, total_timesteps=total_timesteps, max_episode_steps=max_episode_steps, model_path=model_path)
