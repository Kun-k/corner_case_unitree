import json
import os
import numpy as np
from stable_baselines3 import SAC
from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer, TerrainGymEnv

# TODO 功能待验证


def evaluate_policy(
    model_path,
    go2_cfg,
    terrain_cfg,
    episodes=20,
    max_episode_steps=350,
    out_jsonl="failure_chains_sac.jsonl",
):
    trainer = TerrainTrainer(go2_cfg, terrain_cfg)
    env = TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)
    model = SAC.load(model_path)

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for ep in range(episodes):
            obs, info = env.reset()
            ep_chain = []
            has_failure = False

            for _ in range(max_episode_steps):
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = env.step(action)

                transition = {
                    "obs": np.asarray(obs, dtype=np.float32).tolist(),
                    "action": np.asarray(action, dtype=np.float32).tolist(),
                    "reward": float(reward),
                    "next_obs": np.asarray(next_obs, dtype=np.float32).tolist(),
                    "done": bool(terminated or truncated),
                    "terrain_control": np.asarray(action, dtype=np.float32).tolist(),
                    "info": {
                        "fallen": bool(info.get("fallen", False)),
                        "collided": bool(info.get("collided", False)),
                        "base_collision": bool(info.get("base_collision", False)),
                        "thigh_collision": bool(info.get("thigh_collision", False)),
                        "terrain_reward": float(info.get("terrain_reward", 0.0)),
                    },
                }
                ep_chain.append(transition)

                if transition["info"]["fallen"] or transition["info"]["collided"]:
                    has_failure = True

                obs = next_obs
                if terminated or truncated:
                    break

            if has_failure:
                fout.write(json.dumps({"episode": ep, "chain": ep_chain}, ensure_ascii=True) + "\n")

    trainer.close_viewer()


if __name__ == "__main__":
    evaluate_policy(
        model_path="sac_model.zip",
        go2_cfg=["terrain", "go2.yaml"],
        terrain_cfg="train_SAC/terrain_config.yaml",
        episodes=10,
        max_episode_steps=350,
        out_jsonl="train_SAC/failure_chains_sac.jsonl",
    )

