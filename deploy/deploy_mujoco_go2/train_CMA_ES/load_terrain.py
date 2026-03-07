import numpy as np
from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer


if __name__ == "__main__":

    go2_cfg = ["terrain", "go2.yaml"]
    terrain_cfg = "train_CMA_ES/terrain_config.yaml"
    path = "deploy/deploy_mujoco_go2/train_CMA_ES/logs/cmaes_logs/best_angle_array.npy"
    max_robot_steps = 350

    trainer = TerrainTrainer(go2_cfg, terrain_cfg)
    angle_array = np.load(path)

    trainer.init_skip_time = 0
    trainer.init_skip_frame = 10

    trainer.reset()

    trainer.terrain_changer.generate_trig_terrain(angle_array)
    trainer.terrain_changer.enforce_safe_spawn_area(
        center_world=(0.0, 0.0),
        safe_radius_m=1.0,
        blend_radius_m=2.0,
        target_height=0.0,
    )

    if trainer.render:
        trainer.viewer.update_hfield(trainer.terrain_changer.hfield_id)
        trainer.viewer.sync()

    total_reward = 0.0
    done = False

    for _ in range(max_robot_steps):
        _, _, reward, done, _ = trainer.step_only_robot()

        total_reward += float(reward)
        if done:
            break

    # maximize reward -> CMA-ES minimization fitness
    fitness = -total_reward

    print("Total reward: ", total_reward)

    trainer.close_viewer()
