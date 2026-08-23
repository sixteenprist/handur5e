# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""
MDP 观测函数。

坐标系约定:
  (W) = 世界坐标系 (Isaac Sim world, Z up)
  (B) = base_link_1 局部坐标系
  关节空间 = rad (非空间坐标系)
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv


# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════

_FINGERTIP_NAMES = ["thumb4", "index4", "middle4", "ring4", "little4"]

# TCP offset — base_link_1 局部坐标系 (B)，需与 env_cfg.body_offset 和 reset_events 保持一致
_BODY_OFFSET = torch.tensor([0.03, -0.02, 0.06])  # 与 env_cfg 控制器 body_offset 一致


# ══════════════════════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════════════════════

def _robot_body_position(env: ManagerBasedRLEnv, body_name: str) -> torch.Tensor:
    """(W) 指定 body 在世界坐标系下的位置 → (N, 3)."""
    robot: Articulation = env.scene["robot"]
    body_ids, _ = robot.find_bodies(body_name)
    return robot.data.body_pos_w[:, int(body_ids[0])]


# ══════════════════════════════════════════════════════════════
# Cube（世界坐标系）
# ══════════════════════════════════════════════════════════════

def object_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) Cube 质心位置 → (N, 3)."""
    return env.scene["cube_obj"].data.root_pos_w


def object_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) Cube 朝向（四元数 wxyz）→ (N, 4)."""
    return env.scene["cube_obj"].data.root_quat_w


# ══════════════════════════════════════════════════════════════
# 手掌（世界坐标系）
# ══════════════════════════════════════════════════════════════

def palm_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 手掌 base_link_1 位置 → (N, 3)."""
    return _robot_body_position(env, "base_link_1")


def palm_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 手掌朝向（四元数 wxyz）→ (N, 4)."""
    robot: Articulation = env.scene["robot"]
    palm_id = int(robot.find_bodies("base_link_1")[0][0])
    return robot.data.body_quat_w[:, palm_id]


# ══════════════════════════════════════════════════════════════
# TCP & 相对位置（世界坐标系）
# ══════════════════════════════════════════════════════════════

def tcp_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) TCP 在世界坐标系下的位置。
    TCP = palm_pos(W) + R_palm * _BODY_OFFSET(B).
    """
    robot: Articulation = env.scene["robot"]
    palm_id = int(robot.find_bodies("base_link_1")[0][0])
    palm_pos_w = robot.data.body_pos_w[:, palm_id]
    palm_quat_w = robot.data.body_quat_w[:, palm_id]

    from isaaclab.utils.math import quat_apply
    offset_b = _BODY_OFFSET.to(palm_pos_w.device).unsqueeze(0).expand(palm_pos_w.shape[0], -1)  # (N, 3)
    offset_w = quat_apply(palm_quat_w, offset_b)
    return palm_pos_w + offset_w                                  # (N, 3)


def tcp_to_cube(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) TCP → Cube 质心的向量。Stage 1 核心观测。"""
    return env.scene["cube_obj"].data.root_pos_w - tcp_position(env)  # (N, 3)


def palm_to_cube(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 手掌 → Cube 质心的向量。"""
    return env.scene["cube_obj"].data.root_pos_w - palm_position(env)  # (N, 3)


# ══════════════════════════════════════════════════════════════
# 指尖→Cube 相对向量（世界坐标系）
# ══════════════════════════════════════════════════════════════

def fingertip_to_cube_surface(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 五个指尖 → Cube 表面向量，展平为 15D。
    扣除了 Cube 半边长 3cm，0=指尖贴面。
    """
    robot: Articulation = env.scene["robot"]
    cube_pos = env.scene["cube_obj"].data.root_pos_w               # (N, 3)
    half = 0.035  # Cube 半边长（v20: 6cm→7cm）
    vecs = []
    for name in _FINGERTIP_NAMES:
        body_ids, _ = robot.find_bodies(name)
        tip_pos = robot.data.body_pos_w[:, int(body_ids[0])]       # (N, 3)
        to_center = cube_pos - tip_pos                              # (N, 3) 指尖→质心
        dist = torch.norm(to_center, dim=-1, keepdim=True)          # (N, 1)
        to_surface = to_center * torch.clamp((dist - half) / (dist + 1e-8), min=0.0)  # (N, 3) 指尖→表面
        vecs.append(to_surface)
    return torch.cat(vecs, dim=-1)                                 # (N, 15)


# ══════════════════════════════════════════════════════════════
# 指尖触觉（世界坐标系接触力）
# ══════════════════════════════════════════════════════════════

def fingertip_contact_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 五个指尖净接触力范数 → (N, 5)，单位 N。>0 = 碰到 Cube。
    env.scene.sensors[name].data.force_matrix_w[:, 0, 0, :]
    """
    forces = torch.zeros(env.num_envs, 5, device=env.device)
    sensor_names = ["contact_thumb", "contact_index", "contact_middle", "contact_ring", "contact_little"]
    for i, name in enumerate(sensor_names):
        sensor = env.scene.sensors[name]
        fmat = sensor.data.force_matrix_w                           # (N, B, M, 3)
        if fmat is not None and fmat.numel() > 0:
            fw = fmat[:, 0, 0, :]
            forces[:, i] = torch.norm(fw, dim=-1)
    return forces


# ══════════════════════════════════════════════════════════════
# 关节状态 & 动作（非空间坐标系）
# ══════════════════════════════════════════════════════════════

def joint_pos_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """关节位置相对默认值的偏移 (rad) → (N, 26)."""
    robot: Articulation = env.scene["robot"]
    return robot.data.joint_pos - robot.data.default_joint_pos


def joint_vel_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """关节速度 (rad/s) → (N, 26)."""
    return env.scene["robot"].data.joint_vel


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """上一步动作（动作空间值，非空间坐标系）。"""
    return env.action_manager.action


# ══════════════════════════════════════════════════════════════
# 命令（后续阶段用）
# ══════════════════════════════════════════════════════════════

def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """获取指定命令的当前目标值。"""
    return env.command_manager.get_command(command_name)


# ══════════════════════════════════════════════════════════════
# 指尖绝对位置（内部用，非策略观测）
# ══════════════════════════════════════════════════════════════

def thumb_tip_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 拇指尖位置。"""
    return _robot_body_position(env, "thumb4")


def index_tip_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 食指尖位置。"""
    return _robot_body_position(env, "index4")


def middle_tip_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 中指尖位置。"""
    return _robot_body_position(env, "middle4")


def ring_tip_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 无名指尖位置。"""
    return _robot_body_position(env, "ring4")


def little_tip_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 小指尖位置。"""
    return _robot_body_position(env, "little4")