# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""
Episode 终止条件。

三阶段对应：
  Stage 1: success_stage1  ─── 手掌贴 Cube + 稳定（不要求手指）
  Stage 2: success          ─── + 手指闭合
  Stage 3: success_stage3   ─── + Cube 举升高度 >1.35m

坐标系约定: (W) = 世界坐标系
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv

from .observations import _get_body_id


# ══════════════════════════════════════════════════════════════
# 公共内部工具（供 success 系列复用，数值与原内联实现完全一致）
# ══════════════════════════════════════════════════════════════

def _palm_cube_dist(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 手掌 base_link_1 → Cube 质心欧氏距离 → (N,)."""
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["cube_obj"]
    # v85: body_pos_w→body_link_pos_w（link frame 统一）
    palm = robot.data.body_link_pos_w[:, _get_body_id(env, "base_link_1")]
    return torch.norm(palm - obj.data.root_pos_w, dim=-1)

def _cube_stable(env: ManagerBasedRLEnv, lin_vel_threshold: float, ang_vel_threshold: float) -> torch.Tensor:
    """Cube 线/角速度均低于阈值（稳定）→ (N,) bool。"""
    obj: RigidObject = env.scene["cube_obj"]
    lin_vel = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    ang_vel = torch.norm(obj.data.root_ang_vel_w, dim=-1)
    return (lin_vel < lin_vel_threshold) & (ang_vel < ang_vel_threshold)


def _finger_flex_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    """5 组手指关节绝对均值之和（关节 6..26 按 4 个一组）。"""
    robot: Articulation = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    return sum(
        torch.mean(torch.abs(joint_pos[:, s:e]), dim=1)
        for s, e in [(6, 10), (10, 14), (14, 18), (18, 22), (22, 26)]
    )


def _ensure_obj_init_pos(env: ManagerBasedRLEnv, pos: torch.Tensor) -> None:
    """缓存 Cube 初始位置（首次按全量，供出界判断）。"""
    if not hasattr(env, "_obj_init_pos_w") or env._obj_init_pos_w.shape != pos.shape:
        env._obj_init_pos_w = pos.clone()


# ══════════════════════════════════════════════════════════════
# 物体掉落
# ══════════════════════════════════════════════════════════════

def drill_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) Cube Z < 0.70m 视为掉落。"""
    obj: RigidObject = env.scene["cube_obj"]
    return obj.data.root_pos_w[:, 2] < 0.70


# ══════════════════════════════════════════════════════════════
# 成功条件
# ══════════════════════════════════════════════════════════════

def success_stage1(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 1: (W) 手掌-Cube < 0.10m + 稳定。"""
    return (_palm_cube_dist(env) < 0.10) & _cube_stable(env, 0.05, 0.10)


def success_stage2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 2 终止: (W) 手掌-Cube < 0.06m + 稳定。比 Stage 1 更严，防秒终止。"""
    return (_palm_cube_dist(env) < 0.06) & _cube_stable(env, 0.05, 0.10)


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 2: (W) 手掌-Cube < 0.15m + 手指 flex > 0.5 + 稳定。"""
    return (
        (_palm_cube_dist(env) < 0.15)
        & (_finger_flex_term(env) > 0.5)
        & _cube_stable(env, 0.05, 0.10)
    )


def success_stage3(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 3: (W) 抓稳 + Cube Z > 1.35m。"""
    obj: RigidObject = env.scene["cube_obj"]
    return (
        (_palm_cube_dist(env) < 0.15)
        & (_finger_flex_term(env) > 0.5)
        & (obj.data.root_pos_w[:, 2] > 1.35)
        & _cube_stable(env, 0.05, 0.10)
    )


# ══════════════════════════════════════════════════════════════
# 超时
# ══════════════════════════════════════════════════════════════

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episoode 超时（框架自动管理）。"""
    return env.episode_length_buf >= env.max_episode_length - 1


# ══════════════════════════════════════════════════════════════
# 物体出界 / 远离
# ══════════════════════════════════════════════════════════════

def object_out_of_bounds(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    use_xy_only: bool = True,
) -> torch.Tensor:
    """物体相对 episode 初始位置偏移过大即终止。
    use_xy_only=True: 只看水平面偏移 (防推飞)。
    """
    obj: RigidObject = env.scene["cube_obj"]
    pos = obj.data.root_pos_w                                                # (N,3)

    _ensure_obj_init_pos(env, pos)
    init = env._obj_init_pos_w                                                # (N,3)

    if use_xy_only:
        dist = torch.norm(pos[:, :2] - init[:, :2], dim=-1)                   # (N,)
    else:
        dist = torch.norm(pos - init, dim=-1)                                  # (N,)
    return dist > threshold


def cache_object_init_pos_on_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """缓存物体初始位置，供 object_out_of_bounds 使用。"""
    obj: RigidObject = env.scene["cube_obj"]
    pos = obj.data.root_pos_w
    _ensure_obj_init_pos(env, pos)
    env._obj_init_pos_w[env_ids] = pos[env_ids].clone()


def object_away_from_robot(
    env: ManagerBasedRLEnv,
    threshold: float = 2.0,
) -> torch.Tensor:
    """物体距离机器人 root 过远即终止。"""
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["cube_obj"]

    robot_root = robot.data.root_pos_w
    obj_pos = obj.data.root_pos_w
    dist = torch.norm(robot_root - obj_pos, dim=-1)                           # (N,)
    return dist > threshold