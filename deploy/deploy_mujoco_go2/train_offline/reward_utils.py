import os
from typing import Dict, List

import numpy as np
import yaml


def load_reward_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("terrain_config must be dict")
    return cfg.get("event_and_reward", cfg)


def recompute_reward_from_info(info: Dict, reward_cfg: Dict) -> float:
    info = info or {}
    r = 0.0

    if bool(info.get("fallen", False)):
        r += float(reward_cfg.get("fall_reward", 0.0))

    if bool(info.get("base_collision", False)):
        r += float(reward_cfg.get("base_collision_reward", 0.0))

    if bool(info.get("thigh_collision", False)):
        r += float(reward_cfg.get("thigh_collision_reward", 0.0))

    if bool(info.get("collided", False)):
        r += float(reward_cfg.get("collision_reward", 0.0))

    if bool(info.get("stuck", False)):
        r += float(reward_cfg.get("stuck_reward", 0.0))

    # Optional continuous terms if available in logged info.
    if "tilt" in info:
        r += float(reward_cfg.get("tilt_reward_scale", 0.0)) * float(info.get("tilt", 0.0))

    if "speed" in info:
        target_speed = float(reward_cfg.get("target_speed", 0.0))
        speed = float(info.get("speed", 0.0))
        speed_loss = max(0.0, target_speed - speed)
        r += float(reward_cfg.get("speed_reward_scale", 0.0)) * speed_loss

    return float(r)


def recompute_rewards(transitions: List[Dict], reward_cfg: Dict) -> np.ndarray:
    rewards = []
    for tr in transitions:
        info = tr.get("info", {})
        rewards.append(recompute_reward_from_info(info, reward_cfg))
    return np.asarray(rewards, dtype=np.float32)


def compute_returns_advantages(rewards: np.ndarray, dones: np.ndarray, values: np.ndarray, gamma: float, gae_lambda: float):
    n = rewards.shape[0]
    returns = np.zeros((n,), dtype=np.float32)
    adv = np.zeros((n,), dtype=np.float32)

    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(n)):
        non_terminal = 1.0 - float(dones[t])
        delta = float(rewards[t]) + gamma * next_value * non_terminal - float(values[t])
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        adv[t] = last_gae
        returns[t] = adv[t] + float(values[t])
        next_value = float(values[t])

    adv_mean = float(np.mean(adv)) if n > 0 else 0.0
    adv_std = float(np.std(adv)) + 1e-8
    adv = (adv - adv_mean) / adv_std
    return returns.astype(np.float32), adv.astype(np.float32)

