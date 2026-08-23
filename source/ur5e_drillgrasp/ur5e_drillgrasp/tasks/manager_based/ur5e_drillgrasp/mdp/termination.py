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
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv


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
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["cube_obj"]

    palm = robot.data.body_pos_w[:, robot.find_bodies("base_link_1")[0][0]]  # (W)
    cube_pos = obj.data.root_pos_w                                              # (W)
    palm_dist = torch.norm(palm - cube_pos, dim=-1)
    lin_vel = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    ang_vel = torch.norm(obj.data.root_ang_vel_w, dim=-1)

    return (palm_dist < 0.10) & (lin_vel < 0.05) & (ang_vel < 0.10)


def success_stage2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 2 终止: (W) 手掌-Cube < 0.06m + 稳定。比 Stage 1 更严，防秒终止。"""
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["cube_obj"]

    palm = robot.data.body_pos_w[:, robot.find_bodies("base_link_1")[0][0]]  # (W)
    cube_pos = obj.data.root_pos_w                                              # (W)
    palm_dist = torch.norm(palm - cube_pos, dim=-1)
    lin_vel = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    ang_vel = torch.norm(obj.data.root_ang_vel_w, dim=-1)

    return (palm_dist < 0.06) & (lin_vel < 0.05) & (ang_vel < 0.10)


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 2: (W) 手掌-Cube < 0.15m + 手指 flex > 0.5 + 稳定。"""
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["cube_obj"]

    palm = robot.data.body_pos_w[:, robot.find_bodies("base_link_1")[0][0]]  # (W)
    cube_pos = obj.data.root_pos_w                                              # (W)
    palm_dist = torch.norm(palm - cube_pos, dim=-1)
    lin_vel = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    ang_vel = torch.norm(obj.data.root_ang_vel_w, dim=-1)

    joint_pos = robot.data.joint_pos
    finger_flex = sum(
        torch.mean(torch.abs(joint_pos[:, s:e]), dim=1)
        for s, e in [(6,10), (10,14), (14,18), (18,22), (22,26)]
    )

    return (palm_dist < 0.15) & (finger_flex > 0.5) & (lin_vel < 0.05) & (ang_vel < 0.10)


def success_stage3(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 3: (W) 抓稳 + Cube Z > 1.35m。"""
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["cube_obj"]

    palm = robot.data.body_pos_w[:, robot.find_bodies("base_link_1")[0][0]]  # (W)
    cube_pos = obj.data.root_pos_w                                              # (W)
    palm_dist = torch.norm(palm - cube_pos, dim=-1)
    cube_z = obj.data.root_pos_w[:, 2]                                         # (W) Z
    lin_vel = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    ang_vel = torch.norm(obj.data.root_ang_vel_w, dim=-1)

    joint_pos = robot.data.joint_pos
    finger_flex = sum(
        torch.mean(torch.abs(joint_pos[:, s:e]), dim=1)
        for s, e in [(6,10), (10,14), (14,18), (18,22), (22,26)]
    )

    return (palm_dist < 0.15) & (finger_flex > 0.5) & (cube_z > 1.35) & (lin_vel < 0.05) & (ang_vel < 0.10)


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

    if not hasattr(env, "_obj_init_pos_w") or env._obj_init_pos_w.shape != pos.shape:
        env._obj_init_pos_w = pos.clone()
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
    if not hasattr(env, "_obj_init_pos_w") or env._obj_init_pos_w.shape != pos.shape:
        env._obj_init_pos_w = pos.clone()
    else:
        env._obj_init_pos_w[env_ids] = pos[env_ids].clone()


def object_away_from_robot(
    env: ManagerBasedRLEnv,
    threshold: float = 2.0,
) -> torch.Tensor:
    """物体距离机器人 root 过远即终止。"""
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["cube_obj"]

    robot_root = robot.data.root_pos_w
    obj_pos = obj.data.root_pos_w
    dist = torch.norm(robot_root - obj_pos, dim=-1)                           # (N,)
    return dist > threshold