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


# ══════════════════════════════════════════════════════════════
# 超时
# ══════════════════════════════════════════════════════════════

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode 超时（框架自动管理）。"""
    return env.episode_length_buf >= env.max_episode_length - 1