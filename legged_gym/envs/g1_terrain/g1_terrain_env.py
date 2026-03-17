from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os
from math import pi

import yaml

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from isaacgym.torch_utils import quat_apply, torch_rand_float

from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.terrain import Terrain


class G1TerrainRobot(G1Robot):
    """G1 terrain task.

    Reuses G1 robot dynamics/reward implementation while using
    a terrain-oriented config.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.runtime_cfg = self._load_runtime_cfg()
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _plan_spawn_positions(self):
        """Pre-sample per-env spawn positions with zero initial XY noise."""
        max_init_level = self.cfg.terrain.num_rows - 1
        self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
        # Evenly map env ids to terrain type columns.
        self.terrain_types = torch.div(
            torch.arange(self.num_envs, device=self.device) * self.cfg.terrain.num_cols,
            self.num_envs,
            rounding_mode='floor',
        ).to(torch.long)
        self.max_terrain_level = self.cfg.terrain.num_rows

        terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        self._planned_env_origins = terrain_origins[self.terrain_levels, self.terrain_types]

        spawn_noise_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self._planned_spawn_xy = self._planned_env_origins[:, :2] + spawn_noise_xy

    def _get_env_origins(self):
        """Use planned terrain origins for mesh terrains, grid fallback otherwise."""
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"] and hasattr(self, "_planned_env_origins"):
            self.custom_origins = True
            self.env_origins = self._planned_env_origins.clone()
            self.env_origins[:, 2] = 0.
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def create_sim(self):
        """Create sim with terrain choice controlled by runtime config."""
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params
        )

        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            terrain_choice = int(self.runtime_cfg["terrain"]["terrain_choice"])
            self.terrain = Terrain(self.cfg.terrain, self.num_envs, choice=terrain_choice)
            self._plan_spawn_positions()
            # Flatten exactly under generated initial robot XY positions.
            self.terrain.flatten_world_points_to_height(
                self._planned_spawn_xy.detach().cpu().numpy(),
                target_height=0.0,
                radius_m=0.5,
            )
        if mesh_type == "ground_plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()

        self._create_envs()

    def _create_envs(self):
        """Create envs and use the pre-planned initial spawn positions on mesh terrains."""
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            if self.custom_origins and hasattr(self, "_planned_spawn_xy"):
                pos[:2] = self._planned_spawn_xy[i]
            else:
                pos[:2] = self.env_origins[i, :2]
            pos[:2] += 0 * torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    # def _get_env_origins(self):
    #     """Use terrain origins for heightfield/trimesh; fallback to regular grid otherwise."""
    #     if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
    #         self.custom_origins = True
    #         self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
    #         max_init_level = self.cfg.terrain.num_rows - 1
    #         self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
    #         self.terrain_types = torch.div(
    #             torch.arange(self.num_envs, device=self.device),
    #             (self.num_envs / self.cfg.terrain.num_cols),
    #             rounding_mode='floor'
    #         ).to(torch.long)
    #         self.max_terrain_level = self.cfg.terrain.num_rows
    #         self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
    #         self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
    #     else:
    #         self.custom_origins = False
    #         self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
    #         num_cols = np.floor(np.sqrt(self.num_envs))
    #         num_rows = np.ceil(self.num_envs / num_cols)
    #         xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
    #         spacing = self.cfg.env.env_spacing
    #         self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
    #         self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
    #         self.env_origins[:, 2] = 0.

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _resample_commands(self, env_ids):
        """Runtime-configurable command sampler (fixed or random)."""
        cmd_cfg = self.runtime_cfg["commands"]

        if cmd_cfg["fixed_target_lin_vel"]:
            self.commands[env_ids, 0] = torch.tensor(cmd_cfg["target_lin_vel"][0], device=self.device)
            self.commands[env_ids, 1] = torch.tensor(cmd_cfg["target_lin_vel"][1], device=self.device)
        else:
            self.commands[env_ids, 0] = torch_rand_float(
                self.command_ranges["lin_vel_x"][0],
                self.command_ranges["lin_vel_x"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(
                self.command_ranges["lin_vel_y"][0],
                self.command_ranges["lin_vel_y"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        if self.cfg.commands.heading_command:
            if cmd_cfg["fixed_target_heading"]:
                self.commands[env_ids, 3] = torch.tensor(cmd_cfg["target_heading"], device=self.device)
            else:
                self.commands[env_ids, 3] = torch_rand_float(
                    self.command_ranges["heading"][0],
                    self.command_ranges["heading"][1],
                    (len(env_ids), 1),
                    device=self.device,
                ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = -0.0  # 略低于地面
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("vertices: ", self.terrain.vertices.flatten().shape)
        print("triangles: ", self.terrain.triangles.flatten().shape)
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_stairs(self, gym_env):
        num_steps = 5  # 台阶数量
        step_height = 0.1  # 每个台阶的高度
        step_width = 0.5  # 每个台阶的宽度
        step_depth = 0.3  # 每个台阶的深度

        # 创建楼梯
        for i in range(num_steps):
            # 设置每个台阶的位置
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(i * step_depth, 0, i * step_height)

            # 创建台阶的形状
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            box_asset = self.gym.create_box(self.sim, step_width, step_depth, step_height, asset_options)

            # 添加台阶到环境中
            self.gym.create_actor(gym_env, box_asset, pose, "stair_" + str(i), i, 0)

    def _load_runtime_cfg(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        defaults = {
            "terrain": {"terrain_choice": 0},
            "commands": {
                "fixed_target_lin_vel": False,
                "target_lin_vel": [0.8, 0.0],
                "fixed_target_heading": False,
                "target_heading_deg": 0.0,
            },
        }

        for section, values in defaults.items():
            data.setdefault(section, {})
            for key, val in values.items():
                data[section].setdefault(key, val)

        data["commands"]["target_heading"] = float(data["commands"]["target_heading_deg"]) * pi / 180.0
        return data

