# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR5e + Hand2 电钻抓取环境类。

gym 注册（__init__.py）使用 `isaaclab.envs:ManagerBasedRLEnv` + `env_cfg_entry_point`，
本文件提供直接实例化 / 类型引用的环境类（Isaac Lab manager-based 模板惯例）。
"""

from isaaclab.envs import ManagerBasedRLEnv

from .ur5e_drillgrasp_env_cfg import Ur5eDrillgraspEnvCfg


class Ur5eDrillgraspEnv(ManagerBasedRLEnv):
    """UR5e + Hand2 电钻抓取环境。"""

    cfg: Ur5eDrillgraspEnvCfg
