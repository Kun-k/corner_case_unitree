import os
import sys
import mujoco
import numpy as np
import yaml
from typing import Callable, List, Tuple
from deploy.deploy_mujoco_go2.utils import get_gravity_orientation, quat_to_rpy, pd_control

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from deploy.deploy_mujoco_go2.terrain_params import TerrainChanger
from deploy.deploy_mujoco_go2.velocity.go2_controller import Go2Controller


class TerrainTrainer:
    """Environment wrapper that runs the Go2 controller at a high frequency and
    exposes a lower-frequency API for a terrain-controlling agent.

    This class currently supports bump (implemented) and provides interfaces for
    slide_friction and solref (placeholders that call through to the TerrainChanger).
    """

    def __init__(
        self,
        config_file,
        terrain_types=None,
        terrain_config=None,
        terrain_decimation=20,
        render=False,
        terrain_policy=None,
        terrain_policy_device='cpu',
    ):
        self.go2_controller = Go2Controller(config_file)

        # Load Go2 controller config
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/{config_file}", "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.config["xml_path"])
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.config["simulation_dt"]  # 仿真步长，等于G2的控制步长

        # Terrain setup
        self.terrain_types = terrain_types if terrain_types is not None else ['bump']
        self.terrain_config = terrain_config or {}
        self.terrain_decimation = int(terrain_decimation)
        self._robot_counter = 0

        self.render = render
        # per-episode bookkeeping for terrain rewards
        # if repeat_reward is False (default), collision/fall rewards are given only once per episode
        self._fallen_reported = False
        self._collision_reported = False

        # build action dims (same as before)
        self.action_dims = {}
        total = 0
        if 'bump' in self.terrain_types:
            self.action_dims['bump'] = 4
            total += 4
        if 'slide_friction' in self.terrain_types:
            self.action_dims['slide_friction'] = 1
            total += 1
        if 'solref' in self.terrain_types:
            self.action_dims['solref'] = 1
            total += 1
        self.total_action_dims = total

        # Terrain changer helper (owns terrain policy and mapping)
        self.terrain_changer = TerrainChanger(self.model, self.data, action_dims=self.action_dims, terrain_policy=terrain_policy, terrain_policy_device=terrain_policy_device, terrain_config=self.terrain_config)

        # Use controller's control decimation and PD gains
        self.control_decimation = self.go2_controller.control_decimation

    def reset(self):
        """Reset physics and counters. Returns initial observation (robot-centric)."""
        self.data = mujoco.MjData(self.model)
        # recreate terrain_changer with fresh data reference
        self.terrain_changer = TerrainChanger(self.model, self.data, action_dims=self.action_dims, terrain_policy=self.terrain_changer.terrain_policy, terrain_policy_device=self.terrain_changer.terrain_policy_device, terrain_config=self.terrain_config)
        self._robot_counter = 0
        self._terrain_counter = 0
        # reset per-episode flags
        self._fallen_reported = False
        self._collision_reported = False
        # return initial robot observation
        return self._get_robot_observation()

    def _get_robot_observation(self):
        """Return the robot observation vector (same format used for the policy)."""
        qj = self.data.qpos[7:].copy()
        dqj = self.data.qvel[6:].copy()
        quat = self.data.qpos[3:7].copy()
        lin_vel = self.data.qvel[:3].copy()
        ang_vel = self.data.qvel[3:6].copy()

        qj = (qj - self.go2_controller.default_angles) * self.go2_controller.dof_pos_scale
        dqj = dqj * self.go2_controller.dof_vel_scale
        qj = qj[self.go2_controller.policy2model]
        dqj = dqj[self.go2_controller.policy2model]

        gravity_orientation = get_gravity_orientation(quat)
        lin_vel = lin_vel * self.go2_controller.lin_vel_scale
        ang_vel = ang_vel * self.go2_controller.ang_vel_scale

        obs = np.zeros(self.go2_controller.num_obs, dtype=np.float32)
        if self.go2_controller.num_obs == 48:
            obs[:3] = lin_vel
            obs[3:6] = ang_vel
            obs[6:9] = gravity_orientation
            obs[9:12] = self.go2_controller.cmd * self.go2_controller.cmd_scale
            obs[12 : 12 + self.go2_controller.num_actions] = qj
            obs[12 + self.go2_controller.num_actions : 12 + 2 * self.go2_controller.num_actions] = dqj
            obs[12 + 2 * self.go2_controller.num_actions : 12 + 3 * self.go2_controller.num_actions] = self.go2_controller.action_policy_prev
        elif self.go2_controller.num_obs == 45:
            obs[:3] = ang_vel
            obs[3:6] = gravity_orientation
            obs[6:9] = self.go2_controller.cmd * self.go2_controller.cmd_scale
            obs[9:21] = qj
            obs[21:33] = dqj
            obs[33:45] = self.go2_controller.action_policy_prev
        else:
            raise ValueError(f"Unsupported number of observations: {self.go2_controller.num_obs}")

        return obs

    # ---------- terrain adversary reward helpers ----------
    def _is_fallen(self) -> bool:
        """Detect if the robot has fallen based on base height or large body tilt."""
        try:
            base_z = float(self.data.qpos[2])
        except Exception:
            # if qpos not available, assume not fallen
            return False

        # height threshold (meters)
        height_thresh = float(self.terrain_config.get('fall_height_threshold', 0.6))
        if base_z < height_thresh:
            return True

        # orientation check (roll/pitch)
        try:
            r, p, y = quat_to_rpy(self.data.qpos[3:7])
            ang_thresh = float(self.terrain_config.get('fall_angle_threshold', 0.7))
            if abs(r) > ang_thresh or abs(p) > ang_thresh:
                return True
        except Exception:
            pass

        return False

    def _has_collision(self) -> bool:
        """Detect large external contact forces as a proxy for collisions.

        Tries to read `data.cfrc_ext` if available and checks if any body experiences
        force magnitude above a threshold. If the attribute is unavailable, returns False.
        """
        try:
            cfrc = np.array(self.data.cfrc_ext)  # shape (nbody, 6)
            forces = np.linalg.norm(cfrc[:, :3], axis=1)
            max_force = float(np.max(forces))
            force_thresh = float(self.terrain_config.get('collision_force_threshold', 80.0))
            return max_force > force_thresh
        except Exception:
            # attribute not available or other issue; safe fallback: no collision
            return False

    def compute_terrain_reward(self) -> float:
        """Compute the terrain-agent reward: positive when robot collides or falls.

        Returns a scalar reward and stores diagnostic flags in info dict when called from step().
        Configuration via `self.terrain_config` keys:
          - 'fall_height_threshold' (m), default 0.6
          - 'fall_angle_threshold' (rad), default 0.7
          - 'fall_reward', default 1.0
          - 'collision_force_threshold', default 80.0 (N)
          - 'collision_reward', default 0.5
        """
        fallen = self._is_fallen()
        collided = self._has_collision()

        reward = 0.0
        repeat = bool(self.terrain_config.get('repeat_reward', False))

        # give fall reward
        if fallen and (repeat or not self._fallen_reported):
            r = float(self.terrain_config.get('fall_reward', 1.0))
            reward += r
            self._fallen_reported = True

        # give collision reward
        if collided and (repeat or not self._collision_reported):
            r = float(self.terrain_config.get('collision_reward', 0.5))
            reward += r
            self._collision_reported = True

        return reward, {'fallen': fallen, 'collided': collided}

    def step(self):
        # step开始时先执行地形控制，然后执行多步机器人控制

        # if no terrain_action provided, ask TerrainChanger to select one (policy or random fallback)
        robot_obs = self._get_robot_observation()
        terrain_action = self.terrain_changer.select_action(robot_obs, data=self.data)

        # apply terrain action via TerrainChanger
        self.terrain_changer.apply_action_vector(terrain_action)

        # simulation_dt为一个timestep的时间长度
        # control_decimation为控制机器人的周期时间步数量，simulation_dt * control_decimation为控制机器人的周期时长
        # terrain_decimation表示控制几次机器人后控制一次地形
        # total_sim_steps表示一个地形step包含的总仿真步数
        # 地形的控制周期较长，所以一个step中会多次达到self.control_decimation
        total_sim_steps = int(self.terrain_decimation * self.control_decimation)

        # We'll recompute policy output and PD tau at the beginning of each control interval and
        # apply the same tau for physics_steps_per_control consecutive simulator steps (zero-order hold).
        for sim_i in range(total_sim_steps):
            if self.render:  # TODO 啥意思？
                pass

            if sim_i % self.control_decimation == 0:
                target_dof_pos = self.go2_controller.compute_action(self.data)
                target_dq = np.zeros_like(self.go2_controller.kds)
                q = self.data.qpos[7:]
                dq = self.data.qvel[6:]

                for _ in range(4):
                    tau = pd_control(target_dof_pos, q, self.go2_controller.kps, target_dq, dq, self.go2_controller.kds)
                    self.data.ctrl[:] = tau
                    mujoco.mj_step(self.model, self.data)

        # after sim steps, return robot observation to terrain agent
        obs = self._get_robot_observation()

        # TODO 处理reward和info
        # # terrain adversary reward (expose in info but do not replace robot_reward)
        # terrain_reward, terrain_info = self.compute_terrain_reward()
        # info = {'terrain_reward': float(terrain_reward), **terrain_info}
        info = {}

        return obs, None, False, info

    # ----- data collection / viewer-run helpers -----
    def collect_dataset(self, num_steps: int = 1000) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        dataset = []

        # ensure the mujoco viewer is active during simulation
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            obs = self.reset()
            for t in range(int(num_steps)):

                next_obs, _, done, info = self.step()

                action = None  # TODO action
                reward = None  # TODO reward
                dataset.append((obs, action, reward, next_obs, done))

                obs = next_obs

                # optional viewer sync to keep display responsive
                try:
                    viewer.update_hfield(self.terrain_changer.hfield_id)
                    mujoco.mj_setConst(self.model, self.data)
                    mujoco.mj_forward(self.model, self.data)
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                except Exception:
                    pass

        return dataset


if __name__ == "__main__":
    # small demo: collect a short dataset while running the mujoco viewer
    trainer = TerrainTrainer("velocity/configs/go2.yaml", terrain_types=['bump'], terrain_decimation=20, render=True)
    # collect 50 terrain steps and print summary
    dataset = trainer.collect_dataset(num_steps=50)
    print(f"Collected {len(dataset)} transitions")
