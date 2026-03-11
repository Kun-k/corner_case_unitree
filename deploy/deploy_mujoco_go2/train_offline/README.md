# train_offline

Offline pipeline built from `train_SAC/eval.py` logs.

## What it provides

- Configured log-folder ingestion (`logs_config.yaml`)
- Failure summary aggregation from `failure_summary.csv`
- Reward recomputation from transition `info` using reward-only config
- Offline PPO-style training from logged transitions
- Binary failure classifier training (input: state + action)
- Offline eval with classifier-gated action source: PPO or random

## Files

- `logs_config.yaml`: paths of log folders to process
- `terrain_config.yaml`: reward-only weights used for recomputation
- `train_config.yaml`: training/eval parameters
- `data_io.py`: load CSV/PKL datasets from multiple folders
- `reward_utils.py`: recompute reward from transition `info`
- `aggregate_stats.py`: aggregate failure counts/probabilities
- `train_ppo_offline.py`: offline PPO-style training
- `train_classifier.py`: binary classifier training
- `eval.py`: gated offline evaluation
- `run_all.py`: minimal runner for full pipeline
- `plot_fail_rate_trend.py`: plot fail-rate trend and variance shadow from stored `failure_summary.csv`

## Quick start

```bash
python deploy/deploy_mujoco_go2/train_offline/aggregate_stats.py
python deploy/deploy_mujoco_go2/train_offline/plot_fail_rate_trend.py
python deploy/deploy_mujoco_go2/train_offline/train_ppo_offline.py
python deploy/deploy_mujoco_go2/train_offline/train_classifier.py
python deploy/deploy_mujoco_go2/train_offline/eval.py
```

## Output

Artifacts are stored under:

- `deploy/deploy_mujoco_go2/train_offline/train_logs/<log_name>/ppo`
- `deploy/deploy_mujoco_go2/train_offline/train_logs/<log_name>/classifier`
- `deploy/deploy_mujoco_go2/train_offline/train_logs/<log_name>/eval`
- `deploy/deploy_mujoco_go2/train_offline/offline_states_logs/offline_stats/fail_rate_trend.csv`
- `deploy/deploy_mujoco_go2/train_offline/offline_states_logs/offline_stats/fail_rate_trend.png`
