# replay

Replay saved `pkl` trajectories using `TerrainTrainer`.

## Features

- Load `pkl_paths` from `replay_config.yaml`.
- Auto-discover sibling terrain/go2 yaml files in each pkl directory.
- Print bump/time-step info from those yaml files.
- Replay modes:
  - `terrain_action`:
    - realtime terrain: use stored `terrain_action`/`action` through `TerrainTrainer.step(...)`
    - preset terrain: keep preset terrain fixed and only step robot (`step_only_robot`)
  - `robot_action`: replay low-level `tau` from `info.go2_rollout_trace.actions`
  - `robot_state`: replay `qpos/qvel` from `info.go2_rollout_trace.states`
- Terrain replay policy:
  - `terrain_replay.mode: realtime`: apply terrain action at each transition
  - `terrain_replay.mode: preset`: precompute terrain from the current pkl once,
    then replay robot on that fixed terrain; switching to another pkl recomputes it

## Run

```bash
python deploy/deploy_mujoco_go2/replay/replay.py
```

