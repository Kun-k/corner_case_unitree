import mujoco
import mujoco.viewer
import numpy as np
import os
import yaml


class TerrainChanger:
    def __init__(self, model, data, action_dims=None, config_file=None):
        self.model = model
        self.data = data
        # load default config from yaml on disk
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/{config_file}", "r") as f:
            self.terrain_config = yaml.load(f, Loader=yaml.FullLoader)
        self.config_file = config_file

        self.hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain_hfield")
        self.nrow, self.ncol = model.hfield_nrow[self.hfield_id], model.hfield_ncol[self.hfield_id]
        self.hfield = model.hfield_data.reshape(self.nrow, self.ncol)
        self.geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")

        self.original_hfield = model.hfield_data.copy()

        # action space description (dict): e.g. {'bump':4, 'slide_friction':1}
        self.action_dims = action_dims or {}
        self.total_action_dims = sum(self.action_dims.values()) if self.action_dims else 0

        # These are approximate world sizes for mapping between grid and world coordinates.
        # Prefer scene-provided sizes (geom/hfield size half-extents), otherwise fall back to defaults.
        # try to read geom size (half-extents in MuJoCo)
        half_x = float(self.model.geom_size[self.geom_id][0])
        half_y = float(self.model.geom_size[self.geom_id][1])
        self.terrain_size_x = 2.0 * half_x
        self.terrain_size_y = 2.0 * half_y

        # try to read geom position as center
        self.terrain_center_x = float(self.model.geom_pos[self.geom_id][0])
        self.terrain_center_y = float(self.model.geom_pos[self.geom_id][1])

        self.last_action = np.zeros((self.total_action_dims,), dtype=np.float32)

        self.min_forward_dist = self.terrain_config['min_forward_dist']
        self.max_forward_dist = self.terrain_config['max_forward_dist']
        self.max_lateral = self.terrain_config['max_lateral']
        self.max_bump_height = self.terrain_config['max_bump_height']

        grid_resolution = self.terrain_size_x / self.ncol

        radius_min = self.terrain_config['radius_min']
        radius_max = self.terrain_config['radius_max']
        self.radius_grid_min = radius_min / grid_resolution
        self.radius_grid_max = radius_max / grid_resolution

        no_change_radius = self.terrain_config['no_change_radius']
        self.no_change_radius_grid = no_change_radius / grid_resolution

    def reset(self, mujoco_data):
        self.data = mujoco_data
        self.model.hfield_data[:] = self.original_hfield
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.last_action = np.zeros((self.total_action_dims,), dtype=np.float32)

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step = 0

            # self.generate_trig_terrain([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]])
            # viewer.update_hfield(self.hfield_id)
            # mujoco.mj_setConst(self.model, self.data)
            # mujoco.mj_forward(self.model, self.data)
            # mujoco.mj_step(self.model, self.data)

            while viewer.is_running():
                if step % 20000 == 0:

                    # 测试用
                    # action = ...
                    # action = np.random.uniform(-1.0, 1.0, (self.total_action_dims,))
                    # self.apply_action_vector(action)

                    viewer.update_hfield(self.hfield_id)

                    mujoco.mj_setConst(self.model, self.data)
                    mujoco.mj_forward(self.model, self.data)
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                step += 1

    def _world_to_grid(self, x, y):
        grid_x = int((x - self.terrain_center_x + self.terrain_size_x / 2) / self.terrain_size_x * self.ncol)
        grid_y = int((y - self.terrain_center_y + self.terrain_size_y / 2) / self.terrain_size_y * self.nrow)
        return np.clip(grid_x, 0, self.nrow - 1), np.clip(grid_y, 0, self.ncol - 1)

    def apply_action_vector(self, action):
        """
        Interpret a flat action vector according to self.action_dims and call
        the appropriate setters (set_bump, set_slide_friction, set_solref).
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        idx = 0

        if 'bump' in self.action_dims:
            cx_norm = float(action[idx + 0])
            cy_norm = float(action[idx + 1])
            radius = float(action[idx + 2])
            height = float(action[idx + 3])

            # map normalized cx/cy to world coordinates
            # longitudinal distance ahead
            # prefer robot base pos and velocity if available; otherwise use scene origin
            robot_xy = self.data.qpos[:2].copy()
            lin_vel = self.data.qvel[:2].copy()
            speed = np.linalg.norm(lin_vel)

            # 确定速度方向 dir_f
            if speed > 1e-3:
                dir_f = lin_vel / speed
            else:
                qw, qx, qy, qz = self.data.qpos[3:7]
                yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                dir_f = np.array([np.cos(yaw), np.sin(yaw)])

            # dump 相对坐标圆心[dist, lat]
            dist = self.min_forward_dist + (cx_norm + 1.0) / 2.0 * (self.max_forward_dist - self.min_forward_dist)
            lat = cy_norm * self.max_lateral
            perp = np.array([-dir_f[1], dir_f[0]])
            target_xy = robot_xy + dir_f * dist + perp * lat
            gx, gy = self._world_to_grid(float(target_xy[0]), float(target_xy[1]))

            # scale radius to configured grid units
            radius_scaled = (radius + 1.0) / 2.0 * (self.radius_grid_max - self.radius_grid_min) + self.radius_grid_min
            # scale height to configured max
            height_scaled = float(height) * self.max_bump_height

            # 保护机器人所在区域不被修改
            robot_gx, robot_gy = self._world_to_grid(float(robot_xy[0]), float(robot_xy[1]))
            self.set_bump(
                int(gx),
                int(gy),
                float(radius_scaled),
                float(height_scaled),
                int(robot_gx),
                int(robot_gy)
            )

            idx += self.action_dims['bump']

        if 'slide_friction' in self.action_dims:
            mu = float(action[idx])
            mu_scaled = (mu + 1.0) / 2.0 * 1.5 + 0.1
            self.set_slide_friction(float(mu_scaled))
            idx += self.action_dims['slide_friction']

        if 'solref' in self.action_dims:
            sol = float(action[idx])
            solref_scaled = (sol + 1.0) / 2.0 * 0.5 + 0.01
            self.set_solref(float(solref_scaled))
            idx += self.action_dims['solref']

        # self.last_action = np.asarray(action, dtype=np.float32).reshape(-1)

    def set_bump(self, gx, gy, radius, height, robot_gx, robot_gy):

        for row in range(self.nrow):
            for col in range(self.ncol):

                dx = col - gx
                dy = row - gy
                dist = np.sqrt(dx * dx + dy * dy)

                if dist < radius:

                    # ===== 机器人保护区域 =====
                    dx_r = col - robot_gx
                    dy_r = row - robot_gy
                    dist_robot = np.sqrt(dx_r * dx_r + dy_r * dy_r)

                    if dist_robot < self.no_change_radius_grid:
                        continue  # 不允许修改

                    self.hfield[row, col] = height * np.exp(
                        -dist ** 2 / (2 * radius ** 2)
                    )

    def set_slide_friction(self, mu):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        self.model.geom_friction[geom_id][0] = mu

    def set_solref(self, solref):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        self.model.geom_solref[geom_id] = np.array([solref, 1.0])

    def enforce_safe_spawn_area(self, center_world=(0.0, 0.0), safe_radius_m=1, blend_radius_m=1.0, target_height=0.0):
        """
        Make a safe spawn area around center_world with smooth transition:
        - r <= safe_radius_m: fully flattened to target_height
        - safe_radius_m < r < safe_radius_m + blend_radius_m: smooth blend
        - outside: keep original terrain
        """
        cx, cy = center_world
        gx, gy = self._world_to_grid(cx, cy)

        # meter -> grid (use x/y average to keep isotropic disk)
        grid_res_x = self.terrain_size_x / self.ncol
        grid_res_y = self.terrain_size_y / self.nrow
        grid_res = 0.5 * (grid_res_x + grid_res_y)

        safe_r = safe_radius_m / grid_res
        blend_r = max(blend_radius_m / grid_res, 1e-6)

        rows = np.arange(self.nrow, dtype=np.float32)[:, None]
        cols = np.arange(self.ncol, dtype=np.float32)[None, :]
        dist = np.sqrt((rows - gx) ** 2 + (cols - gy) ** 2)

        # t: 0 in safe core, 1 outside blend ring
        t = np.clip((dist - safe_r) / blend_r, 0.0, 1.0)
        # smoothstep for C1-like continuous transition
        w = t * t * (3.0 - 2.0 * t)

        self.hfield[:, :] = (1.0 - w) * target_height + w * self.hfield[:, :]
        self.model.hfield_data[:] = self.hfield.reshape(-1)

    def generate_bumps_terrain(self, bumps_array, safe_pos, safe_radius):
        ...

    def generate_trig_terrain(self, angle_array):
        angle_array = np.array(angle_array)

        # Current scene resolution.
        nx = int(self.ncol)
        ny = int(self.nrow)
        my, mx, _ = angle_array.shape

        # Uniform phase grids over [0, 2pi).
        gx = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False, dtype=np.float32)
        gy = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False, dtype=np.float32)
        X, Y = np.meshgrid(gx, gy)  # [ny, nx]

        terrain = np.zeros((ny, nx), dtype=np.float32)

        # Use angle-array indices as harmonic IDs.
        for iy in range(my):
            fy = iy + 1
            for ix in range(mx):
                fx = ix + 1
                theta, alt = angle_array[iy][ix]
                terrain += (
                    np.sin(fx * X + theta)
                    + 0.5 * np.cos(fy * Y - theta)
                    + 0.35 * np.sin(fx * X + fy * Y + theta)
                ) * alt

        terrain /= float(max(1, mx * my))

        # Normalize then scale to target amplitude.
        max_abs = float(np.max(np.abs(terrain)))
        if max_abs > 1e-8:
            terrain = terrain / max_abs

        # Write back to hfield.
        self.hfield[:, :] = terrain
        self.model.hfield_data[:] = self.hfield.reshape(-1)

        return terrain


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(f"{os.path.dirname(os.path.realpath(__file__))}/robots/go2/scene_terrain.xml")
    data = mujoco.MjData(model)

    # 动态生成bump
    # terrain_changer = TerrainChanger(model, data, action_dims={'bump':4}, config_file="terrain_config.yaml")
    # terrain_changer.run()

    # 三角函数组合
    terrain_changer = TerrainChanger(model, data, action_dims={}, config_file="terrain_config.yaml")
    angle_array = []
    for i in range(10):
        angle_array.append([])
        for j in range(10):
            angle_array[i].append([np.random.uniform(0, 2 * np.pi), np.random.uniform(-1, 1)])
    terrain_changer.generate_trig_terrain(angle_array)
    terrain_changer.enforce_safe_spawn_area()
    terrain_changer.run()
