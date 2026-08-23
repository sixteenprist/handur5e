# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""
可视化验证脚本：检查 Cube 是否在手掌中心。

运行后，在 viewer 窗口中观察：
- Cube（蓝色方块）是否出现在手掌中心
- 终端会打印手掌位置和 Cube 位置的数值对比
- 按 R 键重置，验证每次重置后 Cube 是否回到手心
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="验证 Cube 是否在手掌中心。")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="环境数量（1 个就够了）。")
parser.add_argument("--task", type=str, default="Template-Ur5e-Drillgrasp-v0", help="任务名称。")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import ur5e_drillgrasp.tasks  # noqa: F401


def main():
    """验证 Cube 初始位置是否在手掌中。"""
    # 强制单环境 + 开启 viewer
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    # 确保 viewer 打开
    env_cfg.viewer.eye = (0.8, -0.8, 0.6)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

    env = gym.make(args_cli.task, cfg=env_cfg)

    print("=" * 60)
    print("  Cube-in-Hand 位置验证")
    print("=" * 60)

    # 第一次 reset
    obs, _ = env.reset()
    env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))

    # 读取位置
    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["cube_obj"]

    palm_ids, palm_names = robot.find_bodies("base_link_1")
    palm_idx = int(palm_ids[0])

    palm_pos = robot.data.body_pos_w[0, palm_idx].cpu().numpy()
    cube_pos = obj.data.root_pos_w[0].cpu().numpy()

    tcp_offset = np.array([0.04, -0.08, 0.08])
    expected_cube_pos = palm_pos + tcp_offset
    error = np.linalg.norm(cube_pos - expected_cube_pos)

    print(f"\n📊 手掌位置 (base_link_1):  ({palm_pos[0]:.4f}, {palm_pos[1]:.4f}, {palm_pos[2]:.4f})")
    print(f"📊 预期 Cube 位置 (手掌 + TCP offset): ({expected_cube_pos[0]:.4f}, {expected_cube_pos[1]:.4f}, {expected_cube_pos[2]:.4f})")
    print(f"📊 实际 Cube 位置:                  ({cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f})")
    print(f"📊 误差: {error:.6f} m")

    if error < 0.01:
        print("\n✅ Cube 位置与预期一致！Cube 在手掌中心！")
    else:
        print(f"\n⚠️  误差 {error:.4f}m，请检查 reset_cube_to_palm 逻辑")

    # 检查手指关节状态
    print(f"\n📊 手指关节状态（重置后默认打开）:")
    joint_names = robot.data.joint_names
    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    for i, name in enumerate(joint_names):
        if any(f in name for f in ["thumb", "index", "middle", "ring", "little"]):
            print(f"    {name}: {joint_pos[i]:+.4f}")

    print("\n" + "=" * 60)
    print("  Viewer 已打开，你可以观察 Cube 是否在手心。")
    print("  按 R 键重置 → 验证每次重置后 Cube 是否回到手心。")
    print("  按 ESC 退出。")
    print("=" * 60)

    # 进入交互循环
    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, rew, terminated, truncated, info = env.step(actions)

            step_count += 1

            # 检测是否发生 reset（episode 结束自动重置）
            if terminated.any() or truncated.any():
                palm_pos2 = robot.data.body_pos_w[0, palm_idx].cpu().numpy()
                cube_pos2 = obj.data.root_pos_w[0].cpu().numpy()
                error2 = np.linalg.norm((palm_pos2 + tcp_offset) - cube_pos2)
                print(f"\n🔄 第 {step_count} 步触发重置，重新验证:")
                print(f"   手掌: ({palm_pos2[0]:.4f}, {palm_pos2[1]:.4f}, {palm_pos2[2]:.4f})")
                print(f"   Cube: ({cube_pos2[0]:.4f}, {cube_pos2[1]:.4f}, {cube_pos2[2]:.4f})")
                print(f"   误差: {error2:.6f} m {'✅' if error2 < 0.01 else '⚠️'}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
