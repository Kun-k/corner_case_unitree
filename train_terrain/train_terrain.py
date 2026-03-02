import time

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # env_cfg.env.num_envs = 120
    env_cfg.terrain.mesh_type = "heightfield"

    # TODO
    '''
    terrain_width * terrain_length 的地形面积，分为 num_rows * num_cols 个格子
    '''
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.terrain_length = 20.  # 8.
    env_cfg.terrain.terrain_width = 20.  # 8.

    # 初始化在中心点
    center_x = env_cfg.terrain.terrain_length / 2
    center_y = env_cfg.terrain.terrain_width / 2
    env_cfg.init_state.pos = [center_x, center_y, 0.5]

    env_cfg.env.test = True
    # env_cfg.env.auto_reset = False
    # env_cfg.env.done_on_fall = True

    for i in range(1000):
        # reset environment
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        obs = env.get_observations()

        # load robot policy
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        robot_policy = ppo_runner.get_inference_policy(device=env.device)

        print("================= Episode {} ==================".format(i))
        for j in range(10000):
            # ================= Robot =================
            actions = robot_policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())

            rewards = rews.detach().cpu().numpy()
            dones = dones.detach().cpu().numpy()
            collision = torch.sum((torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1)*1., dim=1).detach().cpu().numpy()

            print("111111111111111111111")
            time.sleep(5)
            env.terrain.customed_terrain(0)
            env.terrain.heightsamples = env.terrain.height_field_raw
            env._create_heightfield()
            print("222222222222222222222")
            time.sleep(5)

            if dones:
                ...
        print("================= End ==================")


if __name__ == '__main__':
    args = get_args()
    play(args)
