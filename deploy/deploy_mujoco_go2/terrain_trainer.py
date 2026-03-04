import os
import sys
import mujoco
import numpy as np
import yaml
from typing import Tuple
import gym
from gym.spaces import Box
from deploy.deploy_mujoco_go2.utils import get_gravity_orientation, quat_to_rpy, pd_control
import mujoco.viewer
import time

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
        go2_config_file,
        terrain_config_file,
    ):
        self.go2_controller = Go2Controller(go2_config_file[1])

        # Load Go2 controller config
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/{go2_config_file[0]}/configs/{go2_config_file[1]}", "r") as f:
            self.go2_config = yaml.load(f, Loader=yaml.FullLoader)

        # MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.go2_config["xml_path"])
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.go2_config["simulation_dt"]  # 仿真步长，等于G2的控制步长

        # Terrain setup
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/{terrain_config_file}", "r") as f:
            self.terrain_config = yaml.load(f, Loader=yaml.FullLoader)
        self.terrain_decimation = int(self.terrain_config["terrain_decimation"])
        self.terrain_types = self.terrain_config["terrain_types"]

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
        self.terrain_changer = TerrainChanger(self.model, self.data, action_dims=self.action_dims, config_file=terrain_config_file)

        # Use controller's control decimation and PD gains
        self.control_decimation = self.go2_controller.control_decimation

        self.render = self.terrain_config["render"]
        self.lock_camera = self.terrain_config["lock_camera"]
        self.realtime_sim = self.terrain_config["realtime_sim"]
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.azimuth = 0
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 1.5
            self.viewer.cam.lookat[:] = self.data.qpos[:3]

        self.step_counter = 0
        self.robot_counter = 0

        self.start_safe_time = self.terrain_config["start_safe_time"]

    def reset(self):
        """Reset physics and counters. Returns initial observation (robot-centric).

        Accept seed/options for now but accept them to be compatible with Gym API
        """
        # ignore seed/options for now but accept them to be compatible with Gym API
        print("reset")
        mujoco.mj_resetData(self.model, self.data)
        # recreate terrain_changer with fresh data reference
        self.terrain_changer.reset(self.data)
        self.step_counter = 0
        self.robot_counter = 0
        # reset per-episode flags
        self._fallen_reported = False
        self._collision_reported = False
        # return initial robot observation (single value)
        return self.get_terrain_observation()

    def close_viewer(self):
        if self.render:
            self.viewer.close()

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
            obs[12: 12 + self.go2_controller.num_actions] = qj
            obs[12 + self.go2_controller.num_actions: 12 + 2 * self.go2_controller.num_actions] = dqj
            obs[12 + 2 * self.go2_controller.num_actions: 12 + 3 * self.go2_controller.num_actions] = self.go2_controller.action_policy_prev
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
        height_thresh = float(self.terrain_config["fall_height_threshold"])
        if base_z < height_thresh:
            return True

        # orientation check (roll/pitch)
        try:
            r, p, y = quat_to_rpy(self.data.qpos[3:7])
            ang_thresh = float(self.terrain_config["fall_angle_threshold"])
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
            force_thresh = float(self.terrain_config["collision_force_threshold"])
            return max_force > force_thresh
        except Exception:
            # attribute not available or other issue; safe fallback: no collision
            return False

    def compute_terrain_reward(self) -> Tuple[float, dict]:
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
        repeat = bool(self.terrain_config["repeat_reward"])

        # give fall reward
        if fallen and (repeat or not self._fallen_reported):
            r = float(self.terrain_config["fall_reward"])
            reward += r
            self._fallen_reported = True

        # give collision reward
        if collided and (repeat or not self._collision_reported):
            r = float(self.terrain_config["collision_reward"])
            reward += r
            self._collision_reported = True

        return reward, {'fallen': fallen, 'collided': collided}

    def step(self, terrain_action):

        if self.data.time > self.start_safe_time:
            self.step_counter += 1

            # --- [步骤 A] 备份当前机器人状态 ---
            qpos_backup = self.data.qpos.copy()
            qvel_backup = self.data.qvel.copy()
            act_backup = self.data.act.copy()

            # apply terrain action via TerrainChanger and remember it
            self.terrain_changer.apply_action_vector(terrain_action)

            mujoco.mj_setConst(self.model, self.data)

            # --- 还原状态 ---
            self.data.qpos[:] = qpos_backup
            self.data.qvel[:] = qvel_backup
            self.data.act[:] = act_backup

            mujoco.mj_forward(self.model, self.data)
            if self.render:
                self.viewer.update_hfield(self.terrain_changer.hfield_id)
                self.viewer.sync()

        total_sim_steps = int(self.terrain_decimation * self.control_decimation)

        # run robot controller with zero-order hold on tau
        for sim_i in range(total_sim_steps):
            self.robot_counter += 1
            step_start = time.time()
            # at control boundaries compute new target and tau
            if sim_i % int(self.control_decimation) == 0:
                tau = self.go2_controller.compute_tau(self.data)
                self.data.ctrl[:] = tau

            mujoco.mj_step(self.model, self.data)

            if self.render:
                if self.lock_camera:
                    self.viewer.cam.lookat[:] = self.data.qpos[:3]
                self.viewer.sync()

            if self.realtime_sim:  # 只能确保仿真不比真实世界快，但是可能会比真实世界慢，取决于计算开销
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        # compute terrain reward and next obs
        terrain_reward, terrain_info = self.compute_terrain_reward()
        next_terrain_obs = self.get_terrain_observation()

        info = {'terrain_reward': float(terrain_reward), **terrain_info}
        done = False  # TODO 是否需要根据fall和collision判断

        print(f"step_counter: {self.step_counter}, robot_counter: {self.robot_counter}, terrain_reward: {terrain_reward}")

        return next_terrain_obs, np.asarray(terrain_action, dtype=np.float32), float(terrain_reward), done, info

    def get_terrain_observation(self):
        """Return observation for terrain policy: robot-centered observation plus last terrain action."""
        robot_obs = self._get_robot_observation()
        # include previous terrain action as part of observation
        if self.total_action_dims > 0:
            return np.concatenate([robot_obs, self.terrain_changer.last_action.astype(np.float32)])
        return robot_obs


class TerrainGymEnv(gym.Env):
    """Gym wrapper around TerrainTrainer for on-policy RL (terrain agent).

    Observation: terrain observation (robot state + last terrain action)
    Action: flat terrain action vector in [-1,1]^action_dim
    Reward: terrain reward (collision/fall)
    """
    def __init__(self, trainer: 'TerrainTrainer', steps_per_terrain: int = None, max_episode_steps: int = 1000):
        super().__init__()
        self.trainer = trainer
        self.steps_per_terrain = steps_per_terrain or trainer.terrain_decimation
        obs_dim = trainer.get_terrain_observation().shape[0]
        act_dim = trainer.total_action_dims
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        if act_dim > 0:
            self.action_space = Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        else:
            self.action_space = Box(low=-1.0, high=1.0, shape=(0,), dtype=np.float32)
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        # call trainer.reset in a way that is compatible with multiple trainer implementations
        res = self.trainer.reset()

        # Normalize return to (obs, info)
        if isinstance(res, tuple):
            if len(res) == 2:
                obs, info = res
            elif len(res) == 1:
                obs = res[0]
                info = {}
            else:
                # unexpected extra return values: take first two
                obs = res[0]
                info = res[1] if isinstance(res[1], dict) else {}
        else:
            obs = res
            info = {}

        self._step_count = 0
        return obs, info

    def step(self, action):
        next_obs, act, reward, done, info = self.trainer.step(terrain_action=action)
        self._step_count += 1

        # Determine truncation (time limit) vs termination (env terminal)
        truncated = False
        if self._step_count >= self.max_episode_steps:
            truncated = True

        terminated = bool(done)

        # Always return the 5-tuple (obs, reward, terminated, truncated, info)
        return next_obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode='human'):
        # trainer handles rendering via its `render` flag and external viewer
        pass


if __name__ == "__main__":
    ...
