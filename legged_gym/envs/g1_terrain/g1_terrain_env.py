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
        if mesh_type == "ground_plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()

        self._create_envs()

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

