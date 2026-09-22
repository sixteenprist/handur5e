# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""Episode 终止条件：cube 掉落 / 超时。

[2026-09-14 清理] 原 success 系列（stage1/2/3）、object_out_of_bounds、object_away_from_robot、
debug_explosion 及其工具函数已删除——均未被 TerminationsCfg 启用（且阈值过时），
需用时从 git 历史恢复。
"""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv


# ══════════════════════════════════════════════════════════════
# 物体掉落
# ══════════════════════════════════════════════════════════════

def drill_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) Cube Z < 0.0m 视为掉落。"""
    obj: RigidObject = env.scene["cube_obj"]
    return obj.data.root_pos_w[:, 2] < 0.0


def cube_launched(
    env: ManagerBasedRLEnv,
    lin_vel_thresh: float = 1.0,
    ang_vel_thresh: float = 5.0,
) -> torch.Tensor:
    """[2026-09-19 干预7 异常隔离] Cube 被撞飞（速度过大）→ 终止该 episode。

    背景：接触探索期偶发"少数 env 把 cube 撞飞/弹动"，产生极端 return 样本 →
    value 尖峰（1.4~13.8）→ 一次 update 撕裂策略（766/1008/1279 均为此模式）。
    正常抓取/悬停时 cube 速度 <0.2 m/s；1.0 m/s 只在"被撞飞"时触发。
    截断这些 rollout：极端 return 不进入 GAE，value 不会被拉偏。
    """
    obj: RigidObject = env.scene["cube_obj"]
    v = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    w = torch.norm(obj.data.root_ang_vel_w, dim=-1)
    return (v > lin_vel_thresh) | (w > ang_vel_thresh)


def fingertip_force_overload(
    env: ManagerBasedRLEnv,
    force_thresh: float = 3.0,
) -> torch.Tensor:
    """[2026-09-19 干预8] 指尖接触力超载（深穿的直接标志）→ 终止该 episode。

    背景：cube 速度阈值（干预7）只抓"被撞飞"；run 22-31 的崩坏是"接触加深 → 手指深穿/卡住
    → 求解挣扎（collect 5→8s）→ value 渐进恶化 → 崩"。深穿时接触力可达数 N（正常抓取 <1N）。
    截断这些 rollout，防深穿样本污染 value。阈值 3N：正常接触（<1N）不受影响。
    """
    from .observations import fingertip_contact_force
    fmax = fingertip_contact_force(env).max(dim=-1).values          # (N,) 任一指尖接触力
    return fmax > force_thresh


def abnormal_robot_state(
    env: ManagerBasedRLEnv,
    joint_vel_thresh: float = 25.0,
) -> torch.Tensor:
    """[2026-09-20 v3] 关节速度超限（物理爆炸/求解释放）→ 终止该 episode。

    依据 DexSuite 的 abnormal_robot_state（关节速度超过限值 2× 即判物理爆炸并终止 + 惩罚）。
    本机数据：正常动作时关节速度 <5 rad/s，崩坏段可达 10+；阈值 25 只截断爆炸，不误伤。
    """
    robot: Articulation = env.scene["robot"]
    return torch.any(torch.abs(robot.data.joint_vel) > joint_vel_thresh, dim=-1)


def cube_out_of_bounds(
    env: ManagerBasedRLEnv,
    z_min: float = 0.5,
    z_max: float = 1.3,
) -> torch.Tensor:
    """[2026-09-20 v3.2 防爆] cube 高度越界 → 终止该 episode。

    依据 DexSuite object_out_of_bound。run 05-47 在 2826 iter 出现 value loss 2200 爆炸：
    策略学提起时，少数 env 的 cube 被顶飞/弹高（z 方向自由、无上限）→ 极端 return 样本。
    正常抓取/提起范围 z∈[0.75, 1.0]（桌面 0.75 + 提起 ≤10cm）；越界即隔离。
    """
    obj: RigidObject = env.scene["cube_obj"]
    z = obj.data.root_pos_w[:, 2]
    return (z > z_max) | (z < z_min)


# ══════════════════════════════════════════════════════════════
# 超时
# ══════════════════════════════════════════════════════════════

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode 超时（框架自动管理）。"""
    return env.episode_length_buf >= env.max_episode_length - 1