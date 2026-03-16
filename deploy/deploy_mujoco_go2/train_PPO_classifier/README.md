# train_PPO

PPO trainer for terrain adversarial environment (`TerrainTrainer` + `TerrainGymEnv`).

Supports optional classifier-gated action source:
- classifier score > threshold: use PPO action
- else: use random action

During gated training, only PPO-selected transitions are added to rollout buffer.

## Run

```bash
python deploy/deploy_mujoco_go2/train_PPO/train.py
python deploy/deploy_mujoco_go2/train_PPO/eval.py
```

## Notes

- Logs are saved to `deploy/deploy_mujoco_go2/train_PPO/train_logs/<log_name>`.
- Failure transition chains are saved to `collision_failures.pkl`.
- This folder mirrors `train_SAC_PPO` structure; main difference is RL algorithm (PPO here).

