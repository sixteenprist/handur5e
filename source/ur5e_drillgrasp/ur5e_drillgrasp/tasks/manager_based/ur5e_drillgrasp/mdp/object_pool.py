# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""混合形状对象池（Phase 2）：每环境同时存在 cube 与 ball，每集激活一个。

- 激活形状由 ``reset_object_pool`` 事件在每集重置时采样（``env._obj_is_ball``）；
- 未激活的物体被摆到台面下（z=-50，地面为单向碰撞面，不干扰）；
- 观测/奖励/终止统一经本模块读取"当前物体"，避免各处直接访问 ``cube_obj``；
- 非对象池模式（``OBJECT_POOL`` 未设）下，所有访问器等价于直接读 ``cube_obj``，
  与 Phase 1 行为逐位一致（由固定种子观测快照验证）。
"""

from __future__ import annotations

import torch

from . import object_geometry as _geo

PARK_REL_XY = (1.5, -1.5)  # 未激活物体停放位置：本环境角落（地面，远离机器人与桌子）
PARK_Z_OFF: float = 0.5     # 从地面上方 0.5m 放下（落到地面静置）
NOMINAL_XY = (-0.14, 0.18)  # 物体默认水平位置（与场景一致）
TABLE_Z: float = 0.75       # 台面高度
CUBE_HALF: float = 0.03     # 方块半边长


def enabled(env) -> bool:
    """对象池模式（由 EnvCfg 读取 OBJECT_POOL 环境变量后设置）。"""
    return bool(getattr(getattr(env, "cfg", None), "object_pool", False))


def _ensure_state(env) -> None:
    if (not hasattr(env, "_obj_is_ball")) or env._obj_is_ball.shape[0] != env.num_envs:
        env._obj_is_ball = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        r = float(getattr(env.cfg, "ball_radius", 0.03))
        env._obj_size = torch.full((env.num_envs,), r, device=env.device)


def is_ball(env) -> torch.Tensor:
    if not enabled(env):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    _ensure_state(env)
    return env._obj_is_ball


def offset(env):
    """当前物体的表面偏移：非池模式=全局设定；池模式=每环境尺寸（cube 0.03 / ball 半径）。"""
    if not enabled(env):
        return _geo.surface_offset()
    _ensure_state(env)
    return env._obj_size


def active_pos(env) -> torch.Tensor:
    cube = env.scene["cube_obj"].data.root_pos_w
    if not enabled(env):
        return cube
    _ensure_state(env)
    ball = env.scene["ball_obj"].data.root_pos_w
    return torch.where(is_ball(env).unsqueeze(-1), ball, cube)


def active_quat(env) -> torch.Tensor:
    cube = env.scene["cube_obj"].data.root_quat_w
    if not enabled(env):
        return cube
    _ensure_state(env)
    ident = torch.zeros_like(cube)
    ident[:, 0] = 1.0  # 球的朝向无意义 → 单位四元数（与训练分布一致）
    return torch.where(is_ball(env).unsqueeze(-1), ident, cube)


def active_lin_vel(env) -> torch.Tensor:
    cube = env.scene["cube_obj"].data.root_lin_vel_w
    if not enabled(env):
        return cube
    ball = env.scene["ball_obj"].data.root_lin_vel_w
    return torch.where(is_ball(env).unsqueeze(-1), ball, cube)


def active_ang_vel(env) -> torch.Tensor:
    cube = env.scene["cube_obj"].data.root_ang_vel_w
    if not enabled(env):
        return cube
    ball = env.scene["ball_obj"].data.root_ang_vel_w
    return torch.where(is_ball(env).unsqueeze(-1), ball, cube)


def surface_distance(env, p: torch.Tensor) -> torch.Tensor:
    """(N,) 指尖/点到当前物体表面的距离（box 精确 AABB / sphere 径向）。"""
    if not enabled(env):
        return _geo.surface_distance(p, env.scene["cube_obj"].data.root_pos_w)
    _ensure_state(env)
    cube = env.scene["cube_obj"].data.root_pos_w
    ball = env.scene["ball_obj"].data.root_pos_w
    d_box = torch.norm(((p - cube).abs() - CUBE_HALF).clamp(min=0.0), dim=-1)
    d_ball = (torch.norm(p - ball, dim=-1) - env._obj_size).clamp(min=0.0)
    return torch.where(is_ball(env), d_ball, d_box)


def surface_vector(env, p: torch.Tensor) -> torch.Tensor:
    """(N,3) 点→当前物体表面向量（与旧观测同式；box 逐位不变 / sphere 精确）。"""
    if not enabled(env):
        return _geo.surface_vector(p, env.scene["cube_obj"].data.root_pos_w)
    _ensure_state(env)
    cube = env.scene["cube_obj"].data.root_pos_w
    ball = env.scene["ball_obj"].data.root_pos_w
    tc = cube - p
    dc = torch.norm(tc, dim=-1, keepdim=True)
    v_box = tc * torch.clamp((dc - CUBE_HALF) / (dc + 1e-8), min=0.0)
    tb = ball - p
    db = torch.norm(tb, dim=-1, keepdim=True)
    v_ball = tb * torch.clamp((db - env._obj_size.unsqueeze(-1)) / (db + 1e-8), min=0.0)
    return torch.where(is_ball(env).unsqueeze(-1), v_ball, v_box)
