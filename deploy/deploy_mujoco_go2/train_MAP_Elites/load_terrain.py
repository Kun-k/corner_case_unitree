import argparse
import numpy as np

from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer


def main():
    parser = argparse.ArgumentParser(description="Load an elite terrain and run Go2 on it.")
    parser.add_argument("--go2-task", type=str, default="terrain")
    parser.add_argument("--go2-config", type=str, default="go2.yaml")
    parser.add_argument("--terrain-config", type=str, default="train_MAP_Elites/terrain_config.yaml")
    parser.add_argument(
        "--angle-array",
        type=str,
        default="deploy/deploy_mujoco_go2/train_MAP_Elites/logs/map_elites_logs_1/best_angle_array.npy",
    )
    parser.add_argument("--max-robot-steps", type=int, default=350)
    parser.add_argument("--safe-radius-m", type=float, default=1.0)
    parser.add_argument("--blend-radius-m", type=float, default=2.0)
    args = parser.parse_args()

    trainer = TerrainTrainer([args.go2_task, args.go2_config], args.terrain_config)
    trainer.init_skip_time = 0
    trainer.init_skip_frame = 10

    angle_array = np.load(args.angle_array)

    try:
        trainer.reset()

        trainer.terrain_changer.generate_trig_terrain(angle_array)
        trainer.terrain_changer.enforce_safe_spawn_area(
            center_world=(0.0, 0.0),
            safe_radius_m=args.safe_radius_m,
            blend_radius_m=args.blend_radius_m,
            target_height=0.0,
        )

        if trainer.render:
            trainer.viewer.update_hfield(trainer.terrain_changer.hfield_id)
            trainer.viewer.sync()

        total_reward = 0.0
        done = False

        for _ in range(args.max_robot_steps):
            _, _, reward, done, _ = trainer.step_only_robot()
            total_reward += float(reward)
            if done:
                break

        print(f"Total reward: {total_reward:.3f}, done={done}")
    finally:
        trainer.close_viewer()


if __name__ == "__main__":
    main()


"""
python deploy/deploy_mujoco_go2/train_MAP_Elites/load_terrain.py \
  --angle-array deploy/deploy_mujoco_go2/train_MAP_Elites/logs/map_elites_logs/best_angle_array.npy 
"""