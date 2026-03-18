# train_SAC_replay

Hybrid training: offline replay chains + online SAC interaction.

## What it does

- Loads replay chains from `replay_pkl_paths`.
- During sampling, replays chain frame-by-frame.
- At each frame, SAC chooses a terrain action and executes it in simulator.
- Stores online transitions into SAC replay buffer.
- Also preloads offline pkl transitions into replay buffer.
- Keeps the same `consecutive_fail_keep_k` filtering and optional classifier-gate interface as `train_SAC`.
- Uses the same training curve callback as `train_SAC`.

## Config

Edit `train_config.yaml`.

Important fields:

- `replay_pkl_paths`
- `replay_require_trace`
- `consecutive_fail_keep_k`
- `classifier_gate.enabled`

## Run

```bash
python deploy/deploy_mujoco_go2/train_SAC_replay/train.py
```

