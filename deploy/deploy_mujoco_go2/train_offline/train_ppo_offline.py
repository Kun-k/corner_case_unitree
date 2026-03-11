import csv
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from deploy.deploy_mujoco_go2.train_offline.data_io import (
    build_transition_arrays,
    get_log_dirs,
    load_transitions_from_logs,
)
from deploy.deploy_mujoco_go2.train_offline.reward_utils import (
    compute_returns_advantages,
    load_reward_config,
    recompute_rewards,
)


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


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def dist(self, obs):
        mean = self.actor(obs)
        std = torch.exp(self.log_std).clamp(min=1e-4, max=5.0)
        return torch.distributions.Normal(mean, std)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


@dataclass
class LogBook:
    rewards: list
    actor_losses: list
    critic_losses: list
    entropies: list


def _save_curve(values, title, out_path, smooth_window=20):
    if len(values) == 0:
        return
    arr = np.asarray(values, dtype=np.float32)
    fig = plt.figure()
    plt.plot(arr, linewidth=1.0)
    if arr.size >= smooth_window:
        k = np.ones((smooth_window,), dtype=np.float32) / float(smooth_window)
        sm = np.convolve(arr, k, mode="valid")
        plt.plot(np.arange(arr.size - sm.size, arr.size), sm, linewidth=2.0)
    plt.title(title)
    plt.grid(True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_metrics_csv(log_dir: str, logbook: LogBook, filename: str):
    path = os.path.join(log_dir, filename)
    n = max(len(logbook.rewards), len(logbook.actor_losses), len(logbook.critic_losses), len(logbook.entropies))
    if n == 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mean_reward", "actor_loss", "critic_loss", "entropy"])
        for i in range(n):
            writer.writerow([
                i + 1,
                logbook.rewards[i] if i < len(logbook.rewards) else "",
                logbook.actor_losses[i] if i < len(logbook.actor_losses) else "",
                logbook.critic_losses[i] if i < len(logbook.critic_losses) else "",
                logbook.entropies[i] if i < len(logbook.entropies) else "",
            ])


def _save_checkpoint(model, optimizer, epoch: int, log_dir: str, name: str):
    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(ckpt, os.path.join(log_dir, name))


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, "train_config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    configure_torch_runtime(cfg)

    log_name = cfg.get("log_name", "offline_default")
    log_dir = os.path.join(base_dir, "train_logs", log_name, "ppo")
    os.makedirs(log_dir, exist_ok=True)

    logs_cfg_path = os.path.join(base_dir, cfg.get("logs_config", "logs_config.yaml"))
    reward_cfg_path = os.path.join(base_dir, cfg.get("terrain_config", "terrain_config.yaml"))
    reward_cfg = load_reward_config(reward_cfg_path)

    # Save runtime configs for reproducibility.
    with open(os.path.join(log_dir, "train_config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(log_dir, "reward_config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"event_and_reward": reward_cfg}, f, sort_keys=False)

    log_dirs, _ = get_log_dirs(logs_cfg_path)
    transitions = load_transitions_from_logs(log_dirs)
    data = build_transition_arrays(transitions)

    states = data["states"]
    actions = data["actions"]
    dones = data["dones"]

    if states.shape[0] == 0:
        raise RuntimeError("No valid transitions loaded from configured log folders.")

    rewards = recompute_rewards(transitions, reward_cfg)

    obs_dim = int(states.shape[1])
    act_dim = int(actions.shape[1])

    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    model = PPOActorCritic(obs_dim, act_dim, hidden=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["ppo"].get("learning_rate", 3e-4)))

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.float32, device=device)

    with torch.no_grad():
        old_dist = model.dist(states_t)
        old_logprob = old_dist.log_prob(actions_t).sum(dim=-1)
        values = model.value(states_t).detach().cpu().numpy()

    returns_np, adv_np = compute_returns_advantages(
        rewards,
        dones,
        values,
        gamma=float(cfg["ppo"].get("gamma", 0.99)),
        gae_lambda=float(cfg["ppo"].get("gae_lambda", 0.95)),
    )

    returns_t = torch.tensor(returns_np, dtype=torch.float32, device=device)
    adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
    old_logprob_t = old_logprob.detach()

    epochs = int(cfg["ppo"].get("epochs", 200))
    batch_size = int(cfg["ppo"].get("batch_size", states.shape[0]))
    minibatch_size = int(cfg["ppo"].get("minibatch_size", 128))
    clip_range = float(cfg["ppo"].get("clip_range", 0.2))
    value_coef = float(cfg["ppo"].get("value_coef", 0.5))
    entropy_coef = float(cfg["ppo"].get("entropy_coef", 0.01))
    max_grad_norm = float(cfg["ppo"].get("max_grad_norm", 0.5))

    save_every = int(cfg["ppo"].get("save_every_epochs", 10))
    ckpt_every = int(cfg["ppo"].get("checkpoint_every_epochs", 20))
    smooth_window = int(cfg["ppo"].get("smooth_window", 20))

    n = states.shape[0]
    indices = np.arange(n)
    logbook = LogBook([], [], [], [])

    for ep in range(1, epochs + 1):
        np.random.shuffle(indices)

        ep_actor_losses = []
        ep_critic_losses = []
        ep_entropies = []

        for start in range(0, min(n, batch_size), minibatch_size):
            idx = indices[start : start + minibatch_size]
            b_states = states_t[idx]
            b_actions = actions_t[idx]
            b_adv = adv_t[idx]
            b_returns = returns_t[idx]
            b_old_logprob = old_logprob_t[idx]

            dist = model.dist(b_states)
            new_logprob = dist.log_prob(b_actions).sum(dim=-1)
            ratio = torch.exp(new_logprob - b_old_logprob)

            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * b_adv
            actor_loss = -torch.min(surr1, surr2).mean()

            values_pred = model.value(b_states)
            critic_loss = nn.functional.mse_loss(values_pred, b_returns)

            entropy = dist.entropy().sum(dim=-1).mean()
            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            ep_actor_losses.append(float(actor_loss.item()))
            ep_critic_losses.append(float(critic_loss.item()))
            ep_entropies.append(float(entropy.item()))

        logbook.rewards.append(float(np.mean(rewards)))
        logbook.actor_losses.append(float(np.mean(ep_actor_losses)) if ep_actor_losses else 0.0)
        logbook.critic_losses.append(float(np.mean(ep_critic_losses)) if ep_critic_losses else 0.0)
        logbook.entropies.append(float(np.mean(ep_entropies)) if ep_entropies else 0.0)

        if ep % save_every == 0:
            _save_metrics_csv(log_dir, logbook, "metrics_partial.csv")
            _save_curve(logbook.rewards, "Offline PPO Mean Reward", os.path.join(log_dir, "reward_partial.png"), smooth_window)
            _save_curve(logbook.actor_losses, "Actor Loss", os.path.join(log_dir, "actor_loss_partial.png"), smooth_window)
            _save_curve(logbook.critic_losses, "Critic Loss", os.path.join(log_dir, "critic_loss_partial.png"), smooth_window)

        if ep % ckpt_every == 0:
            _save_checkpoint(model, optimizer, ep, log_dir, f"checkpoint_epoch_{ep}.pt")

        if ep % 10 == 0:
            print(f"[ppo_offline][epoch {ep}] actor={logbook.actor_losses[-1]:.4f} critic={logbook.critic_losses[-1]:.4f}")

    _save_checkpoint(model, optimizer, epochs, log_dir, "model_final.pt")
    _save_metrics_csv(log_dir, logbook, "metrics.csv")
    _save_curve(logbook.rewards, "Offline PPO Mean Reward", os.path.join(log_dir, "reward.png"), smooth_window)
    _save_curve(logbook.actor_losses, "Actor Loss", os.path.join(log_dir, "actor_loss.png"), smooth_window)
    _save_curve(logbook.critic_losses, "Critic Loss", os.path.join(log_dir, "critic_loss.png"), smooth_window)

    with open(os.path.join(log_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_transitions": int(n),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "epochs": epochs,
                "device": str(device),
            },
            f,
            indent=2,
        )

    print(f"Saved offline PPO logs and model to: {log_dir}")


if __name__ == "__main__":
    main()

