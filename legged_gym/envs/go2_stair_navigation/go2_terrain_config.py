from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2TerrainCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0, 0, 0.42] # x,y,z [m] z =0.42 # -3.0, 1.5, 0.7
        # pos = [-0.0, 0.0, 0.42] # x,y,z [m] z =0.42 # -3.0, 1.5, 0.7
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    # class domain_rand( LeggedRobotCfg.domain_rand):
    #     push_robots = False
    #     randomize_friction = False
  
    class commands( LeggedRobotCfg.commands ):
        # cmd = [target_x, target_y, target_heading]
        num_commands = 3
        heading_command = False
        use_target_pos_cmd = True
        target_pos_kp = 1.0
        target_reach_radius = 0.20
        auto_heading_from_target = True

        class ranges( LeggedRobotCfg.commands.ranges ):
            # Keep command saturation from parent ranges; redefine for clarity.
            heading = [-3.14, 3.14]
            target_x = [-10.0, 10.0]
            target_y = [-10.0, 10.0]

    class rewards( LeggedRobotCfg.rewards ):
        # TODO 用到了吗

        soft_dof_pos_limit = 0.9
        only_positive_rewards = True # True

        tracking_xy_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_ang_vel_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        target_reach_radius = 0.2
        max_speed_reward = 2

        class scales( LeggedRobotCfg.rewards.scales ):
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            torques = -0.0002
            action_rate = -0.01
            collision = -1.
            dof_pos_limits = -10.0
            tracking_ang_vel = 0.5
            feet_air_time = 1.0
            tracking_xy = 1.0
            speed = 0.2

            termination = 0
            tracking_lin_vel = 0
            orientation = 0
            dof_vel = 0
            dof_acc = 0
            base_height = 0
            feet_stumble = 0
            stand_still = 0

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "trimesh" # "heightfield" or "trimesh" or "ground_plane"
        # num_rows = 1
        # num_cols = 1
        border_size = 25 # 25
        terrain_length = 8. # 8.
        terrain_width = 8. # 8.
        curriculum = True
        horizontal_scale = 0.1 # 0.05 # 自定义
        slope_treshold = 1.5 # 自定义

class GO2TerrainCfgPPO( LeggedRobotCfgPPO ):
    seed = -1
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2_terrain'
