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
    """将 Cube 重置到 TCP 位置（wrist_3_link + R_wrist3 * _BODY_OFFSET）。

    与 rewards/observations/update_tcp_marker 的 TCP 定义完全一致（单一来源 _BODY_OFFSET）。
    v91: 硬编码 offset→导入 _BODY_OFFSET（v81 单一来源教训）；docstring 修正——
    代码一直用 quat_apply 旋转 offset，旧注释"未旋转、仅默认姿态近似正确"是过时的。
    当前未在 env_cfg 激活（Stage 1 用桌面固定 cube）。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene["cube_obj"]

    # TCP 参考点载体 wrist_3_link 在世界坐标系下的位置 (W, link frame)
    tcp_idx = _get_body_id(env, "wrist_3_link")
    tcp_pos = robot.data.body_link_pos_w[env_ids, tcp_idx]       # (E, 3)
    tcp_quat = robot.data.body_link_quat_w[env_ids, tcp_idx]     # (E, 4)

    # TCP offset: 在 wrist_3_link 局部坐标系下的偏移 (B)，旋转到世界系（与 obs/reward 一致）
    from isaaclab.utils.math import quat_apply
    from .observations import _BODY_OFFSET, _get_body_id
    tcp_offset = _BODY_OFFSET.to(tcp_pos.device).unsqueeze(0).expand_as(tcp_pos)
    cube_pos = tcp_pos + quat_apply(tcp_quat, tcp_offset)         # (E, 3) 世界坐标系

    # Cube 朝向：世界坐标系恒等四元数 (W)
    cube_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=tcp_pos.device
    ).unsqueeze(0).expand(len(env_ids), -1)

    # 构造 root state: [pos(3), quat(4), vel(3), ang_vel(3)]
    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, :3] = cube_pos
    root_state[:, 3:7] = cube_quat
    root_state[:, 7:10] = 0.0  # 线速度清零
    root_state[:, 10:13] = 0.0  # 角速度清零

    # 写入仿真
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)


def update_tcp_marker(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    interval: int = 1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("tcp_marker"),
) -> None:
    """[调试] 把 tcp_marker 移到当前 TCP 位置——viewer 里直接看到 TCP 在哪。

    TCP = wrist_3_link(W) + R_wrist3 * _BODY_OFFSET(B)，与 rewards/observations 完全一致
    （都用 body_link_pos_w）。由 EventCfg.update_tcp_marker(mode=interval) 每步调用。
    interval 参数是 interval 事件框架约定（每 N 步调用一次），本函数不用它做逻辑。
    正式训练时可注释该事件省 marker 写入开销。
    """
    from isaaclab.utils.math import quat_apply
    from .observations import _BODY_OFFSET, _get_body_id

    robot: Articulation = env.scene["robot"]
    marker = env.scene[asset_cfg.name]

    tcp_idx = _get_body_id(env, "wrist_3_link")
    tcp_pos_w = robot.data.body_link_pos_w[env_ids, tcp_idx]      # (E, 3) link frame 原点
    tcp_quat_w = robot.data.body_link_quat_w[env_ids, tcp_idx]    # (E, 4)

    offset = _BODY_OFFSET.to(tcp_pos_w.device).unsqueeze(0).expand_as(tcp_pos_w)
    tcp_pos = tcp_pos_w + quat_apply(tcp_quat_w, offset)          # (E, 3)

    # marker 不关心朝向，用恒等四元数
    quat_id = torch.zeros(tcp_pos.shape[0], 4, device=tcp_pos.device)
    quat_id[:, 0] = 1.0
    # AssetBaseCfg（无 class_type）→ XFormPrim：set_world_poses 无 env_ids 参数，
    # 一次写入全部实例位姿。interval 事件的 env_ids 恒为全部 env，tcp_pos 形状已匹配。
    marker.set_world_poses(tcp_pos, quat_id)
