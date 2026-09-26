# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""本环境自定义的事件函数（reset/startup 等）。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_middle_curriculum(
    env: ManagerBasedEnv,
    env_ids,
    j2: float = 0.9,
    j3: float = 0.35,
    noise: float = 0.02,
    joint_names: tuple = ("middle2_joint", "middle3_joint"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # [v13 Round-1 混合分布] mix_ref 比例的环境 reset 在参考捏取附近，其余均匀撒在行程上。
    #   目的（Step0 审计后）：卷到位每步 0.11 vs 伸直 0.03，激励正确但"过渡段样本稀少"+
    #   "评论家没见过足够卷曲态"→ 均值停在伸直。混合分布同时给评论家喂卷曲态、逼策略
    #   处理任意起点。课程收尾：闭拢学会后 mix_ref→0、travel_range→只留 0.35。
    mix_ref: float = 0.5,
    ref_j2_range: tuple = (1.13, 1.22),
    ref_j3_range: tuple = (0.40, 0.45),
    travel_j2_range: tuple = (0.35, 1.10),
) -> None:
    """[v12 方案A·重置课程] 每集开始时把中指 j2/j3 设为“课程起始弯曲度”。

    背景（双稳诊断）：训练里策略能“保持已卷姿态”（ref≈0.9），但确定性均值从
      伸展起点（j2=0.35）不会“发起闭拢”（ref≈0.17、m2≈0.46）。策略只从每集
      出生点学起——出生点一直伸直，就学不会第一脚。
    做法：把“出生时中指弯曲度”作为课程变量逐段退直：
      stage1 j2=0.90 → stage2 0.60 → stage3 0.35(目标伸展态，=资产默认)。
      只在 reset 时改写中指两关节（其余关节保持 reset_robot 的默认+噪声），
      不动资产默认角/动作映射/物理，故每段学到的技能跨段不失效。
    实现：读取 reset_robot 之后的当前关节状态，仅覆写中指 j2/j3（±noise 轻随机），
      再整身写回（velocity 用当前值，避免清掉）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = [asset.joint_names.index(n) for n in joint_names]
    pos = asset.data.joint_pos[env_ids].clone()
    n = len(env_ids)
    dev = pos.device
    if mix_ref < 0:  # 兼容旧行为：固定值 + 噪声
        for k, jid in enumerate(ids):
            val = (j2, j3)[k] if len(ids) == 2 else (j2, j3)[min(k, 1)]
            pos[:, jid] = val + (torch.rand(n, device=dev) * 2.0 - 1.0) * noise
    else:
        u = torch.rand(n, device=dev)
        m = u < mix_ref
        lo, hi = travel_j2_range
        j2v = lo + torch.rand(n, device=dev) * (hi - lo)
        lo, hi = ref_j2_range
        j2v[m] = lo + torch.rand(int(m.sum()), device=dev) * (hi - lo)
        # j3 随 j2 线性映射（参考姿态 0.3491→0.4363 对应 1.2217）
        j3v = 0.3491 + (j2v - 0.3491) * ((0.4363 - 0.3491) / (1.2217 - 0.3491))
        lo, hi = ref_j3_range
        j3v[m] = lo + torch.rand(int(m.sum()), device=dev) * (hi - lo)
        pos[:, ids[0]] = j2v + (torch.rand(n, device=dev) * 2.0 - 1.0) * noise
        pos[:, ids[1]] = j3v + (torch.rand(n, device=dev) * 2.0 - 1.0) * noise
    vel = asset.data.joint_vel[env_ids].clone()
    asset.write_joint_state_to_sim(pos, vel, env_ids=env_ids)


def reset_thumb_curriculum(
    env: ManagerBasedEnv,
    env_ids,
    mix: float = 0.5,
    pose: tuple = (-0.70, -1.24, 0.79, 0.70),
    noise: float = 0.05,
    joint_names: tuple = ("thumb1_joint", "thumb2_joint", "thumb3_joint", "thumb4_joint"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """[v19d 拇指 reset 课程] 以 mix 比例把拇指预置到"能捏住"的成功姿态 ± 噪声。

    背景：多段碰撞下用户实测——(t1,t2,t3,t4)=(-0.70,-1.24,0.79,0.70) 时
      拇指尖 0.76N + 中指 0.37N 同时接触（双接触）。策略搜索不到该区域 →
      用 reset 课程喂"拇指已摆好"的状态经验（同中指混合 reset 思路）。
    收尾：学会后 mix=0（回最终使用分布=默认拇指出发）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = [asset.joint_names.index(n) for n in joint_names]
    pos = asset.data.joint_pos[env_ids].clone()
    n = len(env_ids)
    dev = pos.device
    u = torch.rand(n, device=dev)
    m = u < mix
    for k, jid in enumerate(ids[: len(pose)]):
        vals = torch.full((n,), float(pose[k]), device=dev)
        vals = vals + (torch.rand(n, device=dev) * 2.0 - 1.0) * noise
        pos[:, jid] = torch.where(m, vals, pos[:, jid])
    vel = asset.data.joint_vel[env_ids].clone()
    asset.write_joint_state_to_sim(pos, vel, env_ids=env_ids)


def reset_arm_far(
    env: ManagerBasedEnv,
    env_ids,
    scale: float = 0.4,
    noise: float = 0.05,
    joint_names: tuple = ("shoulder_lift_joint", "elbow_joint"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """[v28 接近阶段] 手臂远起：肩抬 +0.25*scale、肘 −0.25*scale（实测 scale=1 时 TCP
    距 cube ~11.6cm 且高于 cube 安全；OSC 目标仍为标定悬停位 → 会自然拉回=天然接近）。

    课程：scale 0.2(≈2cm) → 0.4 → 0.7 → 1.0(≈11.6cm) → 1.3(≈15cm)，达标逐档放大。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = [asset.joint_names.index(n) for n in joint_names]
    pos = asset.data.joint_pos[env_ids].clone()
    n = len(env_ids)
    dev = pos.device
    if len(ids) >= 1:
        pos[:, ids[0]] += 0.25 * scale + (torch.rand(n, device=dev) * 2 - 1) * noise
    if len(ids) >= 2:
        pos[:, ids[1]] += -0.25 * scale + (torch.rand(n, device=dev) * 2 - 1) * noise
    vel = asset.data.joint_vel[env_ids].clone()
    asset.write_joint_state_to_sim(pos, vel, env_ids=env_ids)
