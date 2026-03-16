from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1TerrainCfg(G1RoughCfg):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.9] # x,y,z [m]
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

    class terrain(G1RoughCfg.terrain):
        mesh_type = "trimesh"  # "heightfield" | "trimesh" | "plane"
        # num_rows = 1
        # num_cols = 1
        border_size = 25
        terrain_length = 4.0
        terrain_width = 4.0
        curriculum = True
        horizontal_scale = 0.1
        slope_treshold = 1.5


class G1TerrainCfgPPO(G1RoughCfgPPO):
    seed = 42
    class runner(G1RoughCfgPPO.runner):
        run_name = ""
        experiment_name = "g1_terrain"

