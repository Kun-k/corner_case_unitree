from deploy.deploy_mujoco_go2.train_SAC.eval_random_policy import evaluate_random_policy


if __name__ == "__main__":
    evaluate_random_policy(
        go2_cfg=["terrain", "go2.yaml"],
        terrain_cfg="train_SAC_dense/terrain_config.yaml",
        episodes=10,
        max_episode_steps=350,
        out_jsonl="train_SAC_dense/failure_chains_random.jsonl",
        seed=42,
    )

