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
from isaaclab.utils.math import quat_apply


# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════

_FINGERTIP_NAMES = ["thumb4", "index4", "middle4", "ring4", "little4"]

# Cube 半边长 (m)：6cm 边长的一半；rewards 也复用本常量
_CUBE_HALF_SIZE = 0.025

# TCP offset — wrist_3_link 局部坐标系 (B)，需与 env_cfg.body_offset 和 reset_events 保持一致
_BODY_OFFSET = torch.tensor([0.0, 0.08, 0.11])  # 与 env_cfg body_offset 保持一致（rewards 复用本常量）


# ══════════════════════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════════════════════

# 缓存 body id（find_bodies 遍历 body 名，纯 Python 开销；robot 对象不变，按 (id, name) 缓存）
# v94 性能优化：每步激活路径 ~15 次 find_bodies（obs 9 + reward 6），全部改走缓存，只首次查找。
_body_id_cache: dict = {}


def _get_body_id(env: ManagerBasedRLEnv, body_name: str) -> int:
    """缓存版 body id 查找。"""
    robot: Articulation = env.scene["robot"]
    key = (id(robot), body_name)
    if key not in _body_id_cache:
        ids, _ = robot.find_bodies(body_name)
        _body_id_cache[key] = int(ids[0])
    return _body_id_cache[key]


def _robot_body_position(env: ManagerBasedRLEnv, body_name: str) -> torch.Tensor:
    """(W) 指定 body 在世界坐标系下的位置 → (N, 3).
    v85: 统一用 body_link_pos_w（link frame 原点）——与 rewards._get_palm_pos 等一致。
    原 body_pos_w 是关节 frame（运动 link 的 frame 在关节处），位置会偏。
    v94: find_bodies→_get_body_id（缓存）。
    """
    robot: Articulation = env.scene["robot"]
    return robot.data.body_link_pos_w[:, _get_body_id(env, body_name)]


def _palm_id(env: ManagerBasedRLEnv) -> int:
    """手掌 base_link_1 的 body id。"""
    return _get_body_id(env, "base_link_1")


# ══════════════════════════════════════════════════════════════
# Cube（世界坐标系）
# ══════════════════════════════════════════════════════════════

def object_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) Cube 质心位置 → (N, 3)."""
    return env.scene["cube_obj"].data.root_pos_w


def object_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) Cube 朝向（四元数 wxyz）→ (N, 4)."""
    return env.scene["cube_obj"].data.root_quat_w


def object_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) Cube 质心线速度 → (N, 3).
    【critic 特权观测】success_stage1 的"稳定"判定依赖 cube 线速度 (<0.10)，
    但 actor 观测中没有速度项 → 交给 critic 做更准的价值估计。
    """
    return env.scene["cube_obj"].data.root_lin_vel_w


def object_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) Cube 角速度 → (N, 3).
    【critic 特权观测】同上，"稳定"判定还要求角速度 < 0.20。
    """
    return env.scene["cube_obj"].data.root_ang_vel_w


# ══════════════════════════════════════════════════════════════
# 手掌（世界坐标系）
# ══════════════════════════════════════════════════════════════

def palm_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 手掌 base_link_1 位置 → (N, 3)."""
    return _robot_body_position(env, "base_link_1")


def palm_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 手掌朝向（四元数 wxyz）→ (N, 4).
    v85: body_quat_w→body_link_quat_w（与位置统一用 link frame）。
    """
    robot: Articulation = env.scene["robot"]
    return robot.data.body_link_quat_w[:, _palm_id(env)]


# ══════════════════════════════════════════════════════════════
# TCP & 相对位置（世界坐标系）
# ══════════════════════════════════════════════════════════════

def _tcp_id(env: ManagerBasedRLEnv) -> int:
    """wrist_3_link（TCP 参考点载体）的 body id。"""
    return _get_body_id(env, "wrist_3_link")


def tcp_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) TCP 在世界坐标系下的位置。
    TCP = wrist_3_link_pos(W) + R_wrist3 * _BODY_OFFSET(B).
    v83b: 用 body_link_pos_w/body_link_quat_w（link frame 原点）——用户实测 offset 是相对
    wrist_3_link 的 link 原点；原 body_pos_w 是关节 frame（wrist_3_joint 在 link 根部），
    两者不同导致 TCP 观测偏移。与 rewards._get_tcp_pos 一致。
    """
    robot: Articulation = env.scene["robot"]
    tcp_body_pos_w = robot.data.body_link_pos_w[:, _tcp_id(env)]
    tcp_body_quat_w = robot.data.body_link_quat_w[:, _tcp_id(env)]

    offset_b = _BODY_OFFSET.to(tcp_body_pos_w.device).unsqueeze(0).expand(tcp_body_pos_w.shape[0], -1)  # (N, 3)
    offset_w = quat_apply(tcp_body_quat_w, offset_b)
    return tcp_body_pos_w + offset_w                              # (N, 3)


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
    [2026-09-14 四指共享] 四指分量为强相关冗余，但**维度保持 5 指不变**（obs 98 维与 checkpoint
      兼容）；奖励端聚合才做"拇指组+四指组"两组化（rewards._thumb_four_split）。
    """
    robot: Articulation = env.scene["robot"]
    cube_pos = env.scene["cube_obj"].data.root_pos_w               # (N, 3)
    half = _CUBE_HALF_SIZE
    vecs = []
    for name in _FINGERTIP_NAMES:
        # v85: body_pos_w→body_link_pos_w（link frame，与 rewards._get_fingertip_pos 一致）
        # v94: find_bodies→_get_body_id（缓存）
        tip_pos = robot.data.body_link_pos_w[:, _get_body_id(env, name)]  # (N, 3)
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
    [2026-09-14 四指共享] 输出接口保持 (N,5) 不变；奖励端聚合按拇指组+四指组
      （rewards._thumb_four_split），此处不做切片。
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