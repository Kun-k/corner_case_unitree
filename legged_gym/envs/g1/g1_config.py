from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            # 核心跟踪奖励（提升权重，强化目标）
            tracking_lin_vel = 2.0  # 提高线速度跟踪奖励，让机器人更专注于跟随目标速度
            tracking_ang_vel = 1.0  # 提高角速度跟踪奖励，增强转向控制

            # 基础运动约束（降低惩罚，增加灵活性）
            lin_vel_z = -0.5  # 大幅降低z轴速度惩罚，允许轻微的上下运动
            ang_vel_xy = -0.02  # 降低xy轴角速度惩罚，允许适度的身体摆动
            orientation = -0.2  # 大幅降低姿态惩罚，让机器人更灵活
            base_height = -2.0  # 大幅降低基高惩罚，保留基本高度约束即可
            feet_swing_height = -2.0  # 大幅降低摆高惩罚，允许腿部正常摆动

            # 关节约束（适度惩罚，平衡灵活性和稳定性）
            dof_acc = -1e-6  # 轻微增加关节加速度惩罚，减少抖动
            dof_vel = -5e-4  # 降低关节速度惩罚，增加运动灵活性
            dof_pos_limits = -2.0  # 降低关节限位惩罚，避免过度约束

            # 存活与接触奖励（提升存活奖励，优化接触惩罚）
            alive = 1.0  # 大幅提升存活奖励，优先保证机器人不摔倒
            contact = 0.1  # 降低接触奖励，避免过度依赖地面接触
            contact_no_vel = -0.1  # 降低无速度接触惩罚，减少不必要的约束

            # 其他约束
            feet_air_time = 0.2  # 增加抬脚时间奖励，鼓励自然步态
            collision = -1.0  # 增加碰撞惩罚，避免机器人碰撞自身/环境
            action_rate = -0.005  # 降低动作变化率惩罚，允许更灵活的动作调整
            hip_pos = -0.3  # 降低髋部位置惩罚，减少对髋部的过度约束

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'

  
