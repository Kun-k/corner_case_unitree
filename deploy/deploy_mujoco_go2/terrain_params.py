import mujoco
import mujoco.viewer
import numpy as np


class TerrainChanger:
    def __init__(self, model, data, stepsize=20000, action_dims=None, terrain_policy=None, terrain_policy_device='cpu', terrain_config=None):
        self.model = model
        self.data = data
        self.hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain_hfield")
        self.nrow, self.ncol = model.hfield_nrow[self.hfield_id], model.hfield_ncol[self.hfield_id]
        self.hfield = model.hfield_data.reshape(self.nrow, self.ncol)
        self.geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")

        # action space description (dict): e.g. {'bump':4, 'slide_friction':1}
        self.action_dims = action_dims or {}
        self.total_action_dims = sum(self.action_dims.values()) if self.action_dims else 0

        # terrain policy (callable(obs)->action_vector or torch.jit path or None)
        self.terrain_policy = None
        self.terrain_policy_device = terrain_policy_device
        self.terrain_policy_is_torch = False
        self.terrain_config = terrain_config or {}

        # try to accept a callable or load a torch.jit model from path
        if terrain_policy is not None:
            # callable
            if callable(terrain_policy):
                self.terrain_policy = terrain_policy
            # string path -> attempt to load as torch.jit
            elif isinstance(terrain_policy, str):
                try:
                    import torch
                    self.terrain_policy = torch.jit.load(terrain_policy, map_location=self.terrain_policy_device)
                    self.terrain_policy.eval()
                    self.terrain_policy_is_torch = True
                except Exception:
                    try:
                        # last-ditch: try torch.load
                        import torch
                        m = torch.load(terrain_policy, map_location=self.terrain_policy_device)
                        # if this is a nn.Module, try to use it as callable
                        if hasattr(m, 'eval'):
                            m.eval()
                            self.terrain_policy = m
                            self.terrain_policy_is_torch = True
                        else:
                            self.terrain_policy = None
                    except Exception:
                        self.terrain_policy = None
            else:
                # unsupported type -> ignore
                self.terrain_policy = None

        # These are approximate world sizes for mapping between grid and world coordinates.
        # Prefer scene-provided sizes (geom/hfield size half-extents), otherwise fall back to defaults.
        try:
            # try to read geom size (half-extents in MuJoCo)
            half_x = float(self.model.geom_size[self.geom_id][0])
            half_y = float(self.model.geom_size[self.geom_id][1])
            self.terrain_size_x = 2.0 * half_x
            self.terrain_size_y = 2.0 * half_y
        except Exception:
            # fallback to configured or hardcoded defaults
            self.terrain_size_x = float(self.terrain_config.get('terrain_size_x', 10.0))
            self.terrain_size_y = float(self.terrain_config.get('terrain_size_y', 10.0))

        # try to read geom position as center
        try:
            self.terrain_center_x = float(self.model.geom_pos[self.geom_id][0])
            self.terrain_center_y = float(self.model.geom_pos[self.geom_id][1])
        except Exception:
            self.terrain_center_x = float(self.terrain_config.get('terrain_center_x', 0.0))
            self.terrain_center_y = float(self.terrain_config.get('terrain_center_y', 0.0))

        self.stepsize = stepsize

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step = 0
            while viewer.is_running():
                if step % self.stepsize == 0:

                    # select terrain action (policy or random fallback)
                    action = self.select_action(data=self.data)
                    # apply the (possibly dict-style or flat) action
                    self.apply_action_vector(action)

                    viewer.update_hfield(self.hfield_id)

                    mujoco.mj_setConst(self.model, self.data)
                    mujoco.mj_forward(self.model, self.data)
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                step += 1

    def _world_to_grid(self, x, y):
        grid_x = int((x - self.terrain_center_x + self.terrain_size_x / 2) / self.terrain_size_x * self.nrow)
        grid_y = int((y - self.terrain_center_y + self.terrain_size_y / 2) / self.terrain_size_y * self.ncol)
        return np.clip(grid_x, 0, self.nrow - 1), np.clip(grid_y, 0, self.ncol - 1)

    def _grid_to_world(self, gx, gy):
        x = (gx / float(self.nrow)) * self.terrain_size_x - self.terrain_size_x / 2 + self.terrain_center_x
        y = (gy / float(self.ncol)) * self.terrain_size_y - self.terrain_size_y / 2 + self.terrain_center_y
        return x, y

    def select_action(self, robot_obs=None, data=None):
        """
        Return a flat action vector in [-1,1]^total_action_dims.
        If a terrain policy is available (callable or torch model), it will be used.
        If it returns a dict, it will be flattened using self.action_dims ordering.
        Otherwise a random action is returned as fallback.
        """
        # build observation if needed
        obs_for_policy = None
        if robot_obs is not None:
            obs_for_policy = robot_obs
        elif data is not None:
            # minimal fallback: build a small observation (base pos, lin vel, ang vel)
            try:
                qj = data.qpos[7:].copy()
                dqj = data.qvel[6:].copy()
                quat = data.qpos[3:7].copy()
                lin_vel = data.qvel[:3].copy()
                ang_vel = data.qvel[3:6].copy()
                obs_for_policy = np.concatenate([lin_vel, ang_vel]).astype(np.float32)
            except Exception:
                obs_for_policy = None

        # No policy -> random
        if self.terrain_policy is None:
            return np.random.uniform(-1.0, 1.0, size=(self.total_action_dims,)).astype(np.float32)

        # callable python policy
        try:
            if callable(self.terrain_policy) and not self.terrain_policy_is_torch:
                out = self.terrain_policy(obs_for_policy)
                return self._normalize_policy_output(out)

            # torch model
            if self.terrain_policy_is_torch:
                try:
                    import torch
                    if obs_for_policy is None:
                        # provide a dummy input if policy expects something
                        inp = torch.zeros((1, 1), dtype=torch.float32, device=self.terrain_policy_device)
                    else:
                        inp = torch.from_numpy(np.asarray(obs_for_policy, dtype=np.float32)).unsqueeze(0).to(self.terrain_policy_device)

                    with torch.no_grad():
                        out = self.terrain_policy(inp)

                    # convert output tensor to numpy
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    if hasattr(out, 'detach'):
                        out = out.detach().cpu().numpy()
                    return self._normalize_policy_output(out)
                except Exception:
                    # on error, fallback to random
                    return np.random.uniform(-1.0, 1.0, size=(self.total_action_dims,)).astype(np.float32)

            # unknown policy type -> fallback
            return np.random.uniform(-1.0, 1.0, size=(self.total_action_dims,)).astype(np.float32)

        except Exception:
            # always safe fallback
            return np.random.uniform(-1.0, 1.0, size=(self.total_action_dims,)).astype(np.float32)

    def _normalize_policy_output(self, out):
        # handle dict-style output
        if isinstance(out, dict):
            vec = []
            for key in self.action_dims:
                val = out.get(key)
                if val is None:
                    # pad zeros
                    vec.extend([0.0] * self.action_dims[key])
                else:
                    arr = np.asarray(val).reshape(-1)
                    # truncate or pad
                    need = self.action_dims[key]
                    if arr.size >= need:
                        vec.extend(arr[:need].tolist())
                    else:
                        vec.extend(arr.tolist() + [0.0] * (need - arr.size))
            ret = np.asarray(vec, dtype=np.float32)
        else:
            # assume flat array-like
            ret = np.asarray(out, dtype=np.float32).reshape(-1)

            if ret.size < self.total_action_dims:
                # pad
                pad = np.zeros((self.total_action_dims - ret.size,), dtype=np.float32)
                ret = np.concatenate([ret, pad])
            elif ret.size > self.total_action_dims:
                ret = ret[: self.total_action_dims]

        # clip to [-1,1]
        ret = np.clip(ret, -1.0, 1.0)
        return ret.astype(np.float32)

    def apply_action_vector(self, action):
        """
        Interpret a flat action vector according to self.action_dims and call
        the appropriate setters (set_bump, set_slide_friction, set_solref).
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # pad/truncate
        if action.size < self.total_action_dims:
            action = np.concatenate([action, np.zeros((self.total_action_dims - action.size,), dtype=np.float32)])
        elif action.size > self.total_action_dims:
            action = action[: self.total_action_dims]

        idx = 0
        if 'bump' in self.action_dims:
            cx_norm = float(action[idx + 0])
            cy_norm = float(action[idx + 1])
            radius = float(action[idx + 2])
            height = float(action[idx + 3])

            # map normalized cx/cy to world coordinates
            # longitudinal distance ahead
            # prefer robot base pos and velocity if available; otherwise use scene origin
            try:
                robot_xy = self.data.qpos[:2].copy()
                lin_vel = self.data.qvel[:2].copy()
                speed = np.linalg.norm(lin_vel)
                if speed > 1e-3:
                    dir_f = lin_vel / speed
                else:
                    qw, qx, qy, qz = self.data.qpos[3:7]
                    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                    dir_f = np.array([np.cos(yaw), np.sin(yaw)])
            except Exception:
                robot_xy = np.array([0.0, 0.0])
                dir_f = np.array([1.0, 0.0])

            dist = 0.3 + (cx_norm + 1.0) / 2.0 * (0.8 - 0.3)
            max_lat = self.terrain_config.get('max_lateral', 0.4)
            lat = cy_norm * max_lat
            perp = np.array([-dir_f[1], dir_f[0]])
            target_xy = robot_xy + dir_f * dist + perp * lat

            gx, gy = self._world_to_grid(float(target_xy[0]), float(target_xy[1]))

            radius_scaled = (radius + 1.0) / 2.0 * 20.0 + 5.0  # [5,25] grid units
            height_scaled = float(height) * 0.3  # [-0.3,0.3] m

            # safety radius
            min_dist = self.terrain_config.get('safety_dist', 0.3)
            world_bump_x, world_bump_y = self._grid_to_world(gx, gy)
            try:
                robot_xy = self.data.qpos[:2].copy()
            except Exception:
                robot_xy = np.array([0.0, 0.0])

            if np.linalg.norm(np.array([world_bump_x, world_bump_y]) - robot_xy) < min_dist:
                shift = (min_dist - np.linalg.norm(np.array([world_bump_x, world_bump_y]) - robot_xy)) + 0.05
                gx_new_wx = world_bump_x + dir_f[0] * shift
                gy_new_wy = world_bump_y + dir_f[1] * shift
                gx, gy = self._world_to_grid(gx_new_wx, gy_new_wy)

            self.set_bump(int(gx), int(gy), float(radius_scaled), float(height_scaled))
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

    def set_bump(self, cx, cy, radius, height):
        """
        在 (cx, cy) 生成一个高斯凸起
        Parameters:
            cx (int): x-coordinate of the bump center (grid index)
            cy (int): y-coordinate of the bump center (grid index)
            radius (float): radius of the bump (in grid units)
            height (float): maximum height of the bump at the center (meters)
        """
        for i in range(self.nrow):
            for j in range(self.ncol):
                dx = i - cx
                dy = j - cy
                dist = np.sqrt(dx * dx + dy * dy)

                if dist < radius:
                    self.hfield[i, j] = height * np.exp(
                        -dist ** 2 / (2 * radius ** 2)
                    )

    def set_slide_friction(self, mu):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        self.model.geom_friction[geom_id][0] = mu

    def set_solref(self, solref):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        self.model.geom_solref[geom_id] = np.array([solref, 1.0])


if __name__ == "__main__":
    # Example usage: provide a policy callable. We attempt to build a lightweight SAC policy
    # if stable_baselines3 is available; otherwise fall back to a simple random policy.
    try:
        import gym
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv

        # simple dummy env whose action space matches the bump action (4 dims)
        class DummyEnv(gym.Env):
            def __init__(self, action_dim=4):
                super().__init__()
                from gym.spaces import Box
                self.observation_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                self.action_space = Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

            def reset(self):
                return np.zeros(self.observation_space.shape, dtype=np.float32)

            def step(self, action):
                return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, {}

        dummy_env = DummyEnv(action_dim=4)
        vec_env = DummyVecEnv([lambda: dummy_env])
        model = SAC('MlpPolicy', vec_env, verbose=0)

        def sac_policy(obs):
            # obs may be None (we'll provide a dummy), or a numpy array
            if obs is None:
                x = np.zeros(dummy_env.observation_space.shape, dtype=np.float32)
            else:
                # flatten or trim obs into observation_space shape
                x = np.zeros(dummy_env.observation_space.shape, dtype=np.float32)
            action, _ = model.predict(x, deterministic=False)
            return action

        policy_callable = sac_policy

    except Exception as e:
        print(f"Error: {e}. Stable-baselines3 not available or failed to initialize. Using random policy instead.")
        # fallback random policy: returns flat action for bump [cx_norm, cy_norm, radius, height]
        def random_policy(obs):
            return np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)

        policy_callable = random_policy

    # build a small scene and run the TerrainChanger with the policy
    model = mujoco.MjModel.from_xml_path("deploy/deploy_mujoco_go2/robots/go2/scene_terrain.xml")
    data = mujoco.MjData(model)
    tc = TerrainChanger(model, data, action_dims={'bump': 4}, terrain_policy=policy_callable)

    tc.run()

