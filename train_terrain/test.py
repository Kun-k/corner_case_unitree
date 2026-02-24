import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch
import torch.nn as nn

# ===============================
# 主函数
# ===============================
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
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
    env_cfg.env.auto_reset = False
    env_cfg.env.done_on_fall = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obs = env.get_observations()

    # robot policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    robot_policy = ppo_runner.get_inference_policy(device=env.device)

    max_total_steps = 2000

    for i in range(max_total_steps):

        # ================= Robot =================
        actions = robot_policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # ================= Terrain =================
        robot_root_states = env.root_states

        # terrain_patch = terrain_actor(obs.detach())
        # apply_terrain_patch(env, terrain_patch, robot_root_states)

        # ================= Terrain Reward =================
        # 方式1：minimax
        terrain_reward = -rews

        # # 方式2：只在倒地给正奖励
        # fallen_bonus = dones.float() * 50.0
        # terrain_reward = terrain_reward + fallen_bonus

        print(f"Step {i} | Robot Reward: {rews.mean().item():.3f} "
              f"| Terrain Reward: {terrain_reward.mean().item():.3f}")

        # ===================== 检测倒地事件（保留原有逻辑） =====================
        if dones:  # TODO 倒地后的处理
            is_fallen = infos.get("is_fallen", False)
            current_episode_steps = infos.get("episode_length", 0)
            if is_fallen:
                print(f"第{i}步：机器人倒地！")
            # if is_fallen or (current_episode_steps < env_cfg.env.max_episode_length):
            #     print(f"第{i}步：机器人倒地！")
                # 可选：倒地后重置地形为初始状态
                # current_roughness = 0.0
                # current_slope = 0.0
                # env.terrain_generator.roughness = current_roughness
                # env.terrain_generator.slope = current_slope
                # env.terrain_generator.update_terrain()

    # 测试结束
    print(f"\n测试结束，总运行步数：{max_total_steps}")


if __name__ == '__main__':
    args = get_args()
    play(args)
