import json
import os
import csv
import pickle
import numpy as np
from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer, TerrainGymEnv


def _build_log_paths(logs_subdir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, logs_subdir)
    os.makedirs(log_dir, exist_ok=True)
    return {
        "log_dir": log_dir,
        "pkl": os.path.join(log_dir, "collision_failures.pkl"),
        "csv": os.path.join(log_dir, "failure_summary.csv"),
    }


def _append_csv_row(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episodes_evaluated",
                "total_failures",
                "collision_failures",
                "fall_failures",
                "base_collision_failures",
                "thigh_collision_failures",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_random_policy(
    go2_cfg,
    terrain_cfg,
    episodes=20,
    max_episode_steps=350,
    logs_subdir="logs/default",
    seed=0,
):

    rng = np.random.default_rng(seed)
    trainer = TerrainTrainer(go2_cfg, terrain_cfg)
    env = TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)

    paths = _build_log_paths(logs_subdir)

    current_path = os.path.dirname(os.path.realpath(__file__))
    go2_cfg_file = os.path.join(current_path, "../", go2_cfg[0], go2_cfg[1])
    terrain_cfg_file = os.path.join("../", terrain_cfg)
    # 复制文件到logs
    os.makedirs(paths["log_dir"], exist_ok=True)
    os.system(f"cp {go2_cfg_file} {paths['log_dir']}")
    os.system(f"cp {terrain_cfg_file} {paths['log_dir']}")

    summary = {
        "episodes_evaluated": 0,
        "total_failures": 0,
        "collision_failures": 0,
        "fall_failures": 0,
        "base_collision_failures": 0,
        "thigh_collision_failures": 0,
    }

    for ep in range(episodes):
        obs, info = env.reset()
        ep_chain = []
        has_failure = False
        has_collision = False
        has_fall = False
        has_base_collision = False
        has_thigh_collision = False

        for step_idx in range(max_episode_steps):
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)

            info_dict = {
                "fallen": bool(info.get("fallen", False)),
                "collided": bool(info.get("collided", False)),
                "base_collision": bool(info.get("base_collision", False)),
                "thigh_collision": bool(info.get("thigh_collision", False)),
                "terrain_reward": float(info.get("terrain_reward", 0.0)),
            }

            transition = {
                "obs": np.asarray(obs, dtype=np.float32).tolist(),
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "reward": float(reward),
                "next_obs": np.asarray(next_obs, dtype=np.float32).tolist(),
                "done": bool(terminated or truncated),
                "terrain_control": np.asarray(action, dtype=np.float32).tolist(),
                "info": info_dict,
            }
            ep_chain.append(transition)

            has_collision = has_collision or info_dict["collided"]
            has_fall = has_fall or info_dict["fallen"]
            has_base_collision = has_base_collision or info_dict["base_collision"]
            has_thigh_collision = has_thigh_collision or info_dict["thigh_collision"]
            has_failure = has_collision or has_fall

            obs = next_obs
            if terminated or truncated:
                break

        summary["episodes_evaluated"] += 1
        if has_failure:
            summary["total_failures"] += 1
        if has_collision:
            summary["collision_failures"] += 1
        if has_fall:
            summary["fall_failures"] += 1
        if has_base_collision:
            summary["base_collision_failures"] += 1
        if has_thigh_collision:
            summary["thigh_collision_failures"] += 1

        if has_failure:
            with open(paths["pkl"], "wb") as pf:
                pickle.dump({"episode": ep, "chain": ep_chain}, pf)

        if summary["episodes_evaluated"] % 100 == 0:
            _append_csv_row(paths["csv"], summary.copy())

    _append_csv_row(paths["csv"], summary.copy())

    trainer.close_viewer()


if __name__ == "__main__":
    evaluate_random_policy(
        go2_cfg=["terrain", "go2.yaml"],
        terrain_cfg="train_SAC/terrain_config.yaml",
        episodes=10,
        max_episode_steps=350,
        logs_subdir="logs/random_eval",
        seed=42,
    )
