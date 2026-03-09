import os
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class DenseTrainingLogger(BaseCallback):
    def __init__(self, out_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.out_dir = out_dir
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        os.makedirs(self.out_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))

        if hasattr(self.model, "logger"):
            log_dict = self.model.logger.name_to_value
            if "train/actor_loss" in log_dict:
                self.actor_losses.append(float(log_dict["train/actor_loss"]))
            if "train/critic_loss" in log_dict:
                self.critic_losses.append(float(log_dict["train/critic_loss"]))
        return True

    def _on_training_end(self) -> None:
        self._save_plot(self.episode_rewards, "Episode Reward", "episode_reward.png")
        if len(self.actor_losses) > 0:
            self._save_plot(self.actor_losses, "Actor Loss", "actor_loss.png")
        if len(self.critic_losses) > 0:
            self._save_plot(self.critic_losses, "Critic Loss", "critic_loss.png")

    def _save_plot(self, values, title: str, filename: str):
        if len(values) == 0:
            return
        fig = plt.figure()
        plt.plot(values)
        plt.title(title)
        plt.grid(True)
        fig.savefig(os.path.join(self.out_dir, filename), dpi=140)
        plt.close(fig)

