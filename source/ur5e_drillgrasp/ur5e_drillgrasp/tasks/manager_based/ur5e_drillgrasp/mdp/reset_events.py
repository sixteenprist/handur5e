# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""
Episode 重置事件函数。

实现「手中起始」curriculum：每次 reset 时将 Cube 直接放置在手掌中心，
让策略无需学习"到达"阶段，只需学习"抓握 + 举升"。
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def reset_cube_to_palm(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """将 Cube 重置到手掌中心位置。

    计算方式: palm_pos (世界坐标系) + TCP offset (base_link_1 局部坐标系)
    注意: 当前简化处理，未旋转 offset，仅在默认姿态下近似正确。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene["cube_obj"]

    # 手掌基座在世界坐标系下的位置 (W)
    palm_ids, _ = robot.find_bodies("base_link_1")
    palm_idx = int(palm_ids[0])
    palm_pos = robot.data.body_pos_w[env_ids, palm_idx]  # (E, 3) 世界坐标系

    # TCP offset: 在 base_link_1 局部坐标系下的偏移 (B)
    tcp_offset = torch.tensor([0.03, -0.02, 0.06], device=palm_pos.device)  # (3,) base_link_1 局部坐标系

    # Cube 目标位置（世界坐标系，近似——未对 offset 做旋转变换）(W)
    cube_pos = palm_pos + tcp_offset  # (E, 3) 世界坐标系（近似）

    # Cube 朝向：世界坐标系恒等四元数 (W)
    cube_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=palm_pos.device
    ).unsqueeze(0).expand(len(env_ids), -1)

    # 构造 root state: [pos(3), quat(4), vel(3), ang_vel(3)]
    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, :3] = cube_pos
    root_state[:, 3:7] = cube_quat
    root_state[:, 7:10] = 0.0  # 线速度清零
    root_state[:, 10:13] = 0.0  # 角速度清零

    # 写入仿真
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)
