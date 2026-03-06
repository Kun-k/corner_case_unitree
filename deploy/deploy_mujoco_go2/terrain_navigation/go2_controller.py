import time
import os
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from deploy.deploy_mujoco_go2.utils import get_gravity_orientation, pd_control, quat_to_heading_w, wrap_to_pi


class Go2Controller:
    def __init__(self, config_file):
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/configs/{config_file}", "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.policy = torch.jit.load(self.config['policy_path'])
        print(f"Loaded policy from {self.config['policy_path']}")

        self.num_obs = self.config["num_obs"]
        self.num_actions = self.config["num_actions"]

        self.action_policy_prev = np.zeros(self.num_actions, dtype=np.float32)

        self.xml_path = self.config["xml_path"]
        self.simulation_duration = self.config["simulation_duration"]
        self.simulation_dt = self.config["simulation_dt"]
        self.lock_camera = self.config["lock_camera"]
        self.control_decimation = self.config["control_decimation"]
        self.policy_decimation = self.config["policy_decimation"]
        self.kps = np.array(self.config["kps"], dtype=np.float32)
        self.kds = np.array(self.config["kds"], dtype=np.float32)
        self.default_angles = np.array(self.config["default_angles"], dtype=np.float32)
        self.lin_vel_scale = self.config["lin_vel_scale"]
        self.ang_vel_scale = self.config["ang_vel_scale"]
        self.dof_pos_scale = self.config["dof_pos_scale"]
        self.dof_vel_scale = self.config["dof_vel_scale"]
        self.action_scale = self.config["action_scale"]

        self.reach_threshold_xy = self.config["reach_threshold_xy"]
        self.switch_interval_s = self.config["switch_interval_s"]
        self.auto_switch = self.config["auto_switch"]
        self.switch_on_reach = self.config["switch_on_reach"]
        self.goal_list = np.array(self.config["goal_list"], dtype=np.float32)
        self.goal_idx = 0
        self.goal_random_range = self.config["random_range"]
        self.heading_stiffness = self.config["heading_stiffness"]

        # TODO 1.5不对，改的话记得跟issasim同步
        # obs_scales_distance_x = 1.0 / max(abs(self.goal_random_range[0][0]), abs(self.goal_random_range[0][1])) * 1.5
        # obs_scales_distance_y = 1.0 / max(abs(self.goal_random_range[1][0]), abs(self.goal_random_range[1][1])) * 1.5

        self.cmd_scale = self.config["cmd_scale"]
        self.cmd = np.zeros(3, dtype=np.float32)
        self.cmd_world = np.zeros(3, dtype=np.float32)
        nav_cfg = self.config.get("navigation", {})
        self.kp_pos = float(nav_cfg.get("kp_pos", 1.0))
        self.kp_heading = float(nav_cfg.get("kp_heading", 0.8))
        self.max_cmd_vx = float(nav_cfg.get("max_cmd_vx", 1.0))
        self.max_cmd_vy = float(nav_cfg.get("max_cmd_vy", 1.0))
        self.max_cmd_wz = float(nav_cfg.get("max_cmd_wz", 1.0))

        self.last_change_goal_time = 0.0

    def reset(self):
        self.action_policy_prev = np.zeros(self.num_actions, dtype=np.float32)

    def _update_cmd(self, d, force_switch=False):
        pos_xy = d.qpos[:2].copy()  # TODO 部署时确认，xy方向与issacsim一致

        if not force_switch:
            switch_flag = False
            distance = np.linalg.norm(self.cmd_world[:2], pos_xy)
            switch_flag = switch_flag or (self.switch_on_reach and distance > self.reach_threshold_xy)
            dtime = d.time - self.last_change_goal_time
            switch_flag = switch_flag or (0 < self.switch_interval_s < dtime)

            if not switch_flag:
                return

        if self.auto_switch:
            target_point_x = np.random.uniform(self.goal_random_range["x"][0], self.goal_random_range["x"][1])
            target_point_y = np.random.uniform(self.goal_random_range["y"][0], self.goal_random_range["y"][1])
            target_point_h = np.random.uniform(self.goal_random_range["heading_deg"][0], self.goal_random_range["heading_deg"][1])
            target_point = np.array([target_point_x, target_point_y, target_point_h], dtype=np.float32)
        else:
            target_point = self.goal_list[self.goal_idx]
            self.goal_idx = (self.goal_idx + 1) % len(self.goal_list)

        target_point[2] = target_point[2] * np.pi / 180.0  # convert heading to radians
        self.cmd_world = target_point

        current_heading = quat_to_heading_w(d.qpos[3:7])
        heading_err = wrap_to_pi(target_point[2] - current_heading)
        self.cmd[2] = np.clip(heading_err * self.heading_stiffness, -1, 1)
        self.cmd[:2] = self.cmd_world[:2] - pos_xy

        self.last_change_goal_time = d.time()

    def get_observation(self, d):
        """Return the robot observation vector (same format used for the policy)."""
        qj = d.qpos[7:].copy()
        dqj = d.qvel[6:].copy()
        quat = d.qpos[3:7].copy()
        lin_vel = d.qvel[:3].copy()
        ang_vel = d.qvel[3:6].copy()

        qj = (qj - self.default_angles) * self.dof_pos_scale
        dqj = dqj * self.dof_vel_scale

        gravity_orientation = get_gravity_orientation(quat)
        lin_vel = lin_vel * self.lin_vel_scale
        ang_vel = ang_vel * self.ang_vel_scale

        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[:3] = lin_vel
        obs[3:6] = ang_vel
        obs[6:9] = gravity_orientation
        obs[9:12] = self.cmd * self.cmd_scale
        obs[12: 12 + self.num_actions] = qj
        obs[12 + self.num_actions: 12 + 2 * self.num_actions] = dqj
        obs[12 + 2 * self.num_actions: 12 + 3 * self.num_actions] = self.action_policy_prev

        return obs

    def compute_action(self, d):
        self._update_cmd(d)
        obs = self.get_observation(d)

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        # policy inference
        action_policy = self.policy(obs_tensor).detach().numpy().squeeze()

        # model action order
        target_dof_pos = action_policy * self.action_scale + self.default_angles
        # policy action order used for next step
        self.action_policy_prev[:] = action_policy

        return target_dof_pos

    def run(self):  # 独立启动仿真运行，仿真文件路径来源于config

        # Load robot model
        m = mujoco.MjModel.from_xml_path(self.xml_path)
        d = mujoco.MjData(m)
        m.opt.timestep = self.simulation_dt

        target_dof_pos = self.default_angles.copy()

        # viewer = mujoco.viewer.launch_passive(m, d)
        with mujoco.viewer.launch_passive(m, d) as viewer:
            viewer.cam.azimuth = 0
            viewer.cam.elevation = -20
            viewer.cam.distance = 1.5
            viewer.cam.lookat[:] = d.qpos[:3]
            # Close the viewer automatically after simulation_duration wall-seconds.

            # 一定要先等几帧，不能马上控制
            counter = 1
            while counter % self.control_decimation != 0:
                tau = pd_control(target_dof_pos, d.qpos[7:], self.kps, np.zeros_like(self.kds), d.qvel[6:], self.kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                counter += 1

            start = time.time()
            while viewer.is_running() and time.time() - start < self.simulation_duration:
                step_start = time.time()

                if self.lock_camera:
                    viewer.cam.lookat[:] = d.qpos[:3] # lock camera focus on the robot base

                if counter % self.control_decimation == 0:
                    target_dof_pos = self.compute_action(d)
                tau = pd_control(target_dof_pos, d.qpos[7:], self.kps, np.zeros_like(self.kds), d.qvel[6:], self.kds)
                d.ctrl[:] = tau
                counter += 1

                mujoco.mj_step(m, d)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    config_file = "go2.yaml"

    ctrl = Go2Controller(config_file)
    ctrl.run()

