# train_SAC_dense

Failure-focused SAC training for terrain adversarial control.

## What it does
- Uses `FailureReplayBuffer` to keep **full episode chains** only when:
  - collision/fall happens, or
  - episode cumulative terrain reward reaches a threshold.
- Uses the same `TerrainTrainer` and `TerrainGymEnv` as standard SAC.

## Files
- `train.py`: dense SAC training entry.
- `dense_replay_buffer.py`: failure-chain filtering replay buffer.
- `callbacks.py`: training curve logger.
- `eval_policy.py`: evaluate trained model and log failure five-tuples.
- `terrain_config.yaml`: dense training config.

## Run
```bash
python deploy/deploy_mujoco_go2/train_SAC_dense/train.py
python deploy/deploy_mujoco_go2/train_SAC_dense/eval_policy.py
```

