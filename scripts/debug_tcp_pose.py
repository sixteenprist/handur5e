# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug script: 打印 TCP 世界坐标 + cube 坐标 + 距离，核对 TCP offset 是否正确。

用法（远程，conda isaacsim5.1）:
    python scripts/debug_tcp_pose.py --task ur5e_drillgrasp --num_envs 1 --steps 300 --log_interval 20

说明:
    - TCP = wrist_3_link link 原点 + R_wrist3 * (0, -0.10, -0.08)（与 env 内 reward/obs 完全一致）
    - 每个 env 上方有一个红色小球 marker 跟随 TCP（env_cfg 的 tcp_marker），
      在 viewer 里直接看 TCP 视觉位置。
    - 输出每列: wrist3_link(link frame 原点) | tcp | palm(base_link_1) | cube | tcp_cube_dist
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Debug TCP pose for ur5e_drillgrasp.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="ur5e_drillgrasp", help="Name of the task.")
parser.add_argument("--steps", type=int, default=300, help="Total steps to simulate.")
parser.add_argument("--log_interval", type=int, default=20, help="Print every N steps.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import ur5e_drillgrasp.tasks  # noqa: F401
from ur5e_drillgrasp.tasks.manager_based.ur5e_drillgrasp.mdp import observations as mdp_obs


def main():
    """Debug TCP pose."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_unwrapped = env.unwrapped

    # 复位一次
    env.reset()

    # 打印表头
    print(
        "step | wrist3_link_pos | tcp_pos | palm(base_link_1) | cube_pos | tcp_cube_dist"
    )

    # 模拟若干步
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env_unwrapped.device)
            env.step(actions)

        if step % args_cli.log_interval == 0:
            robot = env_unwrapped.scene["robot"]
            cube = env_unwrapped.scene["cube_obj"]

            # wrist_3_link link frame 原点
            tcp_ids, _ = robot.find_bodies("wrist_3_link")
            tcp_idx = int(tcp_ids[0])
            wrist3_pos = robot.data.body_link_pos_w[0, tcp_idx]

            # TCP（与 reward/obs 一致的算法）
            tcp_pos = mdp_obs.tcp_position(env_unwrapped)[0]

            # 手掌 base_link_1
            palm_ids, _ = robot.find_bodies("base_link_1")
            palm_idx = int(palm_ids[0])
            palm_pos = robot.data.body_link_pos_w[0, palm_idx]

            # cube 质心
            cube_pos = cube.data.root_pos_w[0]

            dist = torch.norm(tcp_pos - cube_pos).item()

            print(
                f"{step:4d} | "
                f"w3=({wrist3_pos[0]:.3f},{wrist3_pos[1]:.3f},{wrist3_pos[2]:.3f}) | "
                f"tcp=({tcp_pos[0]:.3f},{tcp_pos[1]:.3f},{tcp_pos[2]:.3f}) | "
                f"palm=({palm_pos[0]:.3f},{palm_pos[1]:.3f},{palm_pos[2]:.3f}) | "
                f"cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f},{cube_pos[2]:.3f}) | "
                f"dist={dist:.4f}"
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
