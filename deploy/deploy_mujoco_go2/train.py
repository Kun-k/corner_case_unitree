from terrain_trainer import TerrainTrainer, TerrainGymEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt


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


def plot_training(callback):

    plt.figure()
    plt.plot(callback.episode_rewards)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()

    if len(callback.actor_losses) > 0:
        plt.figure()
        plt.plot(callback.actor_losses)
        plt.title("Actor Loss")
        plt.grid()
        plt.show()

    if len(callback.critic_losses) > 0:
        plt.figure()
        plt.plot(callback.critic_losses)
        plt.title("Critic Loss")
        plt.grid()
        plt.show()


def train_sac(go2_cfg, terrain_cfg,
              total_timesteps=20000,
              max_episode_steps=35,
              model_path="sac_model.zip"):

    def make_env():
        trainer = TerrainTrainer(go2_cfg, terrain_cfg)
        env = TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_env])

    model = SAC("MlpPolicy", vec_env, verbose=1)

    callback = TrainingLoggerCallback()

    model.learn(total_timesteps=total_timesteps,
                callback=callback)

    model.save(model_path)

    plot_training(callback)


if __name__ == "__main__":
    # TODO 参数化
    # TODO 绘图
    go2_cfg = ["terrain", "go2.yaml"]
    terrain_cfg = "terrain_config.yaml"
    total_timesteps = 20000
    max_episode_steps = 35
    model_path = "sac_model.zip"
    train_sac(go2_cfg, terrain_cfg, total_timesteps=total_timesteps, max_episode_steps=max_episode_steps, model_path=model_path)
