# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MDP 奖励函数。

三阶段课程学习对应关系：
  Stage 1 接近 ─── reach, object_distance_penalty
  Stage 2 抓取 ─── + finger_close, fingertip, contact_force
  Stage 3 举升 ─── + lift, palm_height

坐标系约定:
  (W) = 世界坐标系 (Isaac Sim world, Z up)
  (B) = base_link_1 局部坐标系
  (C) = Cube 质心
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════

# 手指关节按名称模式匹配（兼容新 USD 命名，如 thumb4_hoint / 关节顺序变化）
_FINGER_JOINT_PATTERNS = {
    "thumb":  "thumb.*",
    "index":  "index.*",
    "middle": "middle.*",
    "ring":   "ring.*",
    "little": "little.*",
}
# 缓存每根手指的关节 id（find_joints 较慢，robot 不会变）
_finger_joint_ids_cache: dict = {}


def _get_finger_joint_ids(robot: Articulation) -> list[torch.Tensor]:
    key = id(robot)
    if key not in _finger_joint_ids_cache:
        ids_list = []
        for pattern in _FINGER_JOINT_PATTERNS.values():
            ids, _ = robot.find_joints(pattern)
            ids_list.append(torch.as_tensor(ids, device=robot.device))
        _finger_joint_ids_cache[key] = ids_list
    return _finger_joint_ids_cache[key]
# v81: 常量改从 observations 单一来源导入。v71 同步 TCP 偏移时漏改本文件本地副本
# （_BODY_OFFSET 仍为 (0.04,-0.02,0.08)、_CUBE_HALF_SIZE 仍为 0.0375），导致所有奖励
# 的 TCP 距离按错误偏移计算（与真实 TCP 差 ~4cm）：奖励认为"假 TCP 到位"→ 策略继续压
# → 真实手掌还差 4cm → 下降接近时手掌撞 cube（用户 2026-08-27 报告）。单一来源防再不同步。
from .observations import _BODY_OFFSET, _CUBE_HALF_SIZE, _FINGERTIP_NAMES, _get_body_id


# ══════════════════════════════════════════════════════════════
# 基础工具（世界坐标系，供本模块内部使用）
# ══════════════════════════════════════════════════════════════

def _get_palm_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 手掌 base_link_1 位置 → (N,3)."""
    robot: Articulation = env.scene["robot"]
    return robot.data.body_link_pos_w[:, _get_body_id(env, "base_link_1")]


def _get_cube_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) Cube 质心位置 → (N,3)."""
    return env.scene["cube_obj"].data.root_pos_w


def _get_tcp_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) TCP 位置 = wrist_3_link_pos(W) + R_wrist3 * body_offset(B)."""
    robot: Articulation = env.scene["robot"]
    tcp_body_id = _get_body_id(env, "wrist_3_link")
    tcp_body_pos = robot.data.body_link_pos_w[:, tcp_body_id]
    tcp_body_quat = robot.data.body_link_quat_w[:, tcp_body_id]

    from isaaclab.utils.math import quat_apply
    offset = _BODY_OFFSET.to(tcp_body_pos.device).unsqueeze(0).expand(tcp_body_pos.shape[0], -1)
    return tcp_body_pos + quat_apply(tcp_body_quat, offset)


def _tcp_cube_dist(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) TCP → Cube 质心欧氏距离 → (N,)."""
    return torch.norm(_get_tcp_pos(env) - _get_cube_pos(env), dim=-1)


def _palm_cube_dist(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 手掌 → Cube 质心欧氏距离 → (N,).
    [已废弃] Stage 1 改用 _tcp_cube_dist，保留用于 palm_height_reward。"""
    return torch.norm(_get_palm_pos(env) - _get_cube_pos(env), dim=-1)


def _get_fingertip_pos(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """(W) 单个指尖 link 位置 → (N,3)."""
    robot: Articulation = env.scene["robot"]
    return robot.data.body_link_pos_w[:, _get_body_id(env, name)]


def _fingertip_cube_surface_dists(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 五指尖→Cube 表面距离 → (N,5)。扣除了 Cube 半边长。每指独立。"""
    cube = _get_cube_pos(env)
    dists = []
    for t in _FINGERTIP_NAMES:
        d = torch.norm(_get_fingertip_pos(env, t) - cube, dim=-1)
        d = torch.clamp(d - _CUBE_HALF_SIZE, min=0.0)
        dists.append(d.unsqueeze(-1))
    return torch.cat(dists, dim=-1)


# 每指 4 关节权重（v41：四指掌根2 权重升 + 层级弯曲）
# v36: 指尖4 权重 0.8→0.2——不再奖励指尖过度蜷缩
# v41: 四指 掌根2 权重 0.8→1.2——根节先弯、弯得多（用户要求）
# 四指 [侧摆1, 掌根2, 中段3, 指尖4] = [0.2, 1.2, 1.5, 0.2]（掌根2 + 中段3 主导）
# 拇指 [对掌1, 侧摆2, 中段3, 指尖4] = [1.2, 0.2, 1.5, 0.2]（对掌1 保留高权重，它是拇指抓握关键）
# v18 崩溃根因 = 只有掌根(2)弯、中段(3)不弯 → 指尖够不到、只有压力没接触 → 挤飞
_FINGER_JOINT_WEIGHTS = [
    torch.tensor([1.2, 0.2, 1.5, 0.2]),  # 拇指: 对掌1 / 侧摆2 / 中段3 / 指尖4
    torch.tensor([0.2, 1.2, 1.5, 0.2]),  # 食指: 侧摆1 / 掌根2 / 中段3 / 指尖4
    torch.tensor([0.2, 1.2, 1.5, 0.2]),  # 中指
    torch.tensor([0.2, 1.2, 1.5, 0.2]),  # 无名指
    torch.tensor([0.2, 1.2, 1.5, 0.2]),  # 小指
]


def _finger_flex(robot: Articulation) -> torch.Tensor:
    """手指弯曲度量 (rad)。0=伸直, ~1.57=全弯。
    关节加权 + 层级门控（v41）：
    - 四指 掌根2 权重 1.2——根节先弯、弯得多
    - 后续关节在上一关节已弯（>0.15）的基础上才计分：j3 计分需 j2 弯，j4 计分需 j3 弯
      （sigmoid 软门）——自然的手指顺序弯曲，防"指尖独弯的蜷缩爪"
    - 拇指保持原权重，不做层级门控（解剖不同）
    跨指 0.7×最不弯 + 0.3×平均，防拇指独自弯完。
    """
    abs_pos = torch.abs(robot.data.joint_pos)
    per_finger = []
    for i, ids in enumerate(_get_finger_joint_ids(robot)):
        w = _FINGER_JOINT_WEIGHTS[i].to(abs_pos.device)      # (4,)
        q = abs_pos[:, ids]                                   # (N,4) [j1,j2,j3,j4]
        if i > 0 and len(ids) >= 4:
            # 层级门控：j3 计分需 j2 弯 >0.15，j4 计分需 j3 弯 >0.15
            g3 = torch.sigmoid(8.0 * (q[:, 1] - 0.15))
            g4 = torch.sigmoid(8.0 * (q[:, 2] - 0.15))
            contrib = w[0]*q[:, 0] + w[1]*q[:, 1] + w[2]*q[:, 2]*g3 + w[3]*q[:, 3]*g4
            per_finger.append(contrib / w.sum())
        else:
            per_finger.append((q * w).sum(dim=1) / w.sum())
    per_finger = torch.stack(per_finger, dim=0)              # (5, N) 加权平均
    return 0.7 * per_finger.min(dim=0).values + 0.3 * per_finger.mean(dim=0)


# ══════════════════════════════════════════════════════════════
# Dense Reward: 接近与抓取
# ══════════════════════════════════════════════════════════════

def reach_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    sat_dist: float | None = None,
    sat_transition: float = 0.01,
    orient_threshold: float | None = None,
) -> torch.Tensor:
    """(W) TCP→Cube 距离奖励 1−tanh(d/σ)。
    各处梯度均匀，不像高斯在远处几乎平直。
    0.5m→0.04, 0.3m→0.17, 0.2m→0.34, 0.1m→0.62, 0.04m→0.84。

    v82: sat_dist 非 None 时，d < sat_dist 给满分 1.0（饱和）——消除"越近越好"。
    v91: 加 sat_transition——饱和边界平滑过渡（[sat, sat+trans] 内 1→基础值线性），
    避免 sat_dist 硬跳变梯度过尖（默认 0.01 几乎保持原行为，env_cfg 显式传 0.01）。

    v93: 加 orient_threshold 姿态门控——TCP 姿态角 > orient_threshold 时 reach 归零。
    reach 只奖 TCP 距离，翻转/倾斜能通过旋转 _BODY_OFFSET 方向让 TCP 转向 cube（捷径），
    姿态惩罚 -0.05 压不住；门控后翻转+接近=0 vs 平放+接近=满分，翻转无利可图。
    与 success_stage1_reward/tcp_orientation_penalty 同源（_tcp_init_quat_w 由 reset 事件缓存）。
    """
    d = _tcp_cube_dist(env)
    r = 1.0 - torch.tanh(d / std)
    if sat_dist is not None:
        if sat_transition > 0:
            # 过渡带 [sat, sat+trans]：1 → tanh(sat+trans) 值，与 tanh 分支连续
            # （v92 初版用 clamp((d_sat-d)/trans) 在 d=d_sat 处 0→tanh 值跳变，修正为连续）
            d_sat = sat_dist + sat_transition
            r_end = 1.0 - torch.tanh(torch.tensor(d_sat / std, device=d.device))  # 过渡带终点的 tanh 值
            ramp = torch.clamp((d_sat - d) / sat_transition, min=0.0, max=1.0)    # 1→0
            r_sat = r_end + (1.0 - r_end) * ramp                                  # d=sat→1.0, d=d_sat→r_end（连续）
            r = torch.where(d < d_sat, r_sat, r)
        else:
            r = torch.where(d < sat_dist, torch.ones_like(r), r)
    if orient_threshold is not None:
        robot: Articulation = env.scene["robot"]
        tcp_body_id = _get_body_id(env, "wrist_3_link")
        q_curr = robot.data.body_link_quat_w[:, tcp_body_id]
        if not hasattr(env, "_tcp_init_quat_w"):
            env._tcp_init_quat_w = q_curr.clone()
        q_init = env._tcp_init_quat_w
        dot = (q_curr * q_init).sum(dim=-1).abs().clamp(-1.0, 1.0)
        ang = 2.0 * torch.acos(dot)
        oriented = (ang < orient_threshold).float()
        r = r * oriented
    return r


def finger_close_reward(
    env: ManagerBasedRLEnv,
    dist_threshold: float = 0.08,
    max_flex: float = 0.5,
) -> torch.Tensor:
    """硬门控：TCP 距 Cube < dist_threshold 才奖励弯曲，否则 0。
    防手指提前弯、半路把 Cube 撞飞。
    v19: flex 上限 1.0→0.5——防策略无限加压把 cube 挤飞
    （v18@3053 崩溃：峰值握力→cube 被挤出→终止→value loss 爆 57→策略乱抖螺旋）。
    弯到 0.5 即饱和，不再奖励更大力（实测抓握 flex≈0.2，0.5 足够）。
    """
    dist = _tcp_cube_dist(env)                                       # (N,)
    gate = (dist < dist_threshold).float()                            # 硬门：0 或 1
    flex = _finger_flex(env.scene["robot"])
    return gate * torch.min(flex, torch.tensor(max_flex, device=flex.device))


def middle_flex_reward(
    env: ManagerBasedRLEnv,
    dist_threshold: float = 0.10,
    max_flex: float = 0.5,
) -> torch.Tensor:
    """(v24→v25) 直接奖励中段关节3弯曲（食/中/无/小 各组第3个关节）。
    v25 关键改动：聚合 mean→0.5·mean+0.5·min。
    play(2026-08-23) 证实：环+小指中段已弯、食+中指中段不弯、食中指指尖还远。
    纯 mean 会被已弯的环+小指稀释（不弯食中指也能拿一半分）→ 梯度不足。
    min 半权重直接盯"最不弯"的（食+中指），逼策略补齐短板（与 fingertip 混合聚合同款）。
    取各组第3个关节：与 GroupedHandAction 的 action[i]→组内第i关节映射一致，
    play 已证实该顺序正确（环+小指 action 第3维确实驱动中段）。
    """
    robot: Articulation = env.scene["robot"]
    ids = []
    for grp in ("index", "middle", "ring", "little"):
        j, _ = robot.find_joints(grp + ".*")
        if len(j) > 2:
            ids.append(j[2])          # 第3个关节 = 中段
    ids_t = torch.tensor(ids, device=robot.device)
    abs3 = torch.abs(robot.data.joint_pos[:, ids_t])                   # (N,4)
    flex3 = 0.5 * abs3.mean(dim=1) + 0.5 * abs3.min(dim=1).values      # (N,)
    dist = _tcp_cube_dist(env)
    gate = (dist < dist_threshold).float()
    return gate * torch.min(flex3, torch.tensor(max_flex, device=flex3.device))


def base_flex_reward(
    env: ManagerBasedRLEnv,
    dist_threshold: float = 0.10,
    max_flex: float = 0.5,
    fingers: tuple = ("index", "middle", "ring", "little"),
) -> torch.Tensor:
    """(v44→v45) 掌根(关节2)弯曲奖励——让手指靠掌根包住 cube，而不是靠指尖(j4)硬够。
    v44(play 3800)：ring/little 的 j2 弯不够、j4 过弯（指尖独弯去够表面）→ 定向 ring/little。
    v45(play 3907)：四指的 j2 都可以再弯一点（手指整体包络更好）→ 扩展到 index/middle/ring/little。
    掌根先弯 → 整根手指包住 cube → 指尖自然落在侧面，无需 j4 过度蜷缩。
    与 finger_close 区别：它是全指聚合(0.7min+0.3mean)，单指 j2 提升会被 min 稀释，
    没有定向梯度——本项单独给每根指定手指的掌根压力。
    """
    robot: Articulation = env.scene["robot"]
    ids = []
    for grp in fingers:
        j, _ = robot.find_joints(grp + ".*")
        if len(j) > 1:
            ids.append(j[1])          # 第2个关节 = 掌根
    ids_t = torch.tensor(ids, device=robot.device)
    abs2 = torch.abs(robot.data.joint_pos[:, ids_t])                  # (N, len)
    flex2 = 0.5 * abs2.mean(dim=-1) + 0.5 * abs2.min(dim=-1).values   # (N,)
    dist = _tcp_cube_dist(env)
    gate = (dist < dist_threshold).float()
    return gate * torch.min(flex2, torch.tensor(max_flex, device=flex2.device))


def thumb_contact_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.04,
    gate_std: float = 0.15,
) -> torch.Tensor:
    """(v46) 拇指指尖(thumb4)→Cube 表面距离奖励。
    play(4100)：四指已接近到位，但拇指没碰到 cube。
    fingertip_contact 是全指聚合(0.7mean+0.3min)，thumb 的梯度被 mean 稀释，
    拇指够不到时没有定向压力——本项单独奖励拇指指尖贴近 cube 表面。
    """
    dists = _fingertip_cube_surface_dists(env)          # (N,5) 第0列 = thumb
    per = 1.0 - torch.tanh(dists[:, 0] / std)           # (N,)
    tcp_dist = _tcp_cube_dist(env)
    gate = 1.0 - torch.tanh(tcp_dist / gate_std)        # Soft TCP gate
    return per * gate


def fingertip_contact_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.04,
    gate_std: float = 0.15,
) -> torch.Tensor:
    """(W) 指尖→Cube 表面距离奖励 × TCP 门控。
    每指独立 1−tanh(d/σ)，v37 聚合 = 0.7·mean + 0.3·min：
    - mean 半：平滑梯度，每根手指独立可学
    - min 半：保留对"最远那根"（无名指/小指）的温和压力，但 0.5→0.3 降权——
      0.5·min 逼它们把关节3+4弯到极限去够表面 → 蜷缩爪（v36 观察）。
      人手抓取靠中段包络、指尖适度伸直，0.3·min 让它们停在自然姿态。
    v23: std 0.08→0.04——0.08 时差 1-2cm 就 82%，无"最后 1cm"梯度；
    0.04 只奖励真正贴到，恢复接触梯度（策略已贴近，不再怕尖梯度难学）。
    """
    dists = _fingertip_cube_surface_dists(env)               # (N,5)
    per_finger = 1.0 - torch.tanh(dists / std)                # (N,5)
    raw = 0.7 * per_finger.mean(dim=-1) + 0.3 * per_finger.min(dim=-1).values  # (N,)
    # Soft TCP gate
    tcp_dist = _tcp_cube_dist(env)
    gate = 1.0 - torch.tanh(tcp_dist / gate_std)
    return raw * gate


def fingertip_overcurl_penalty(
    env: ManagerBasedRLEnv,
    thresh: float = 0.5,
    dist_threshold: float = 0.15,
) -> torch.Tensor:
    """(v36) 惩罚指尖关节4过度蜷缩——正常人手抓取靠中段包络，指尖应保持适度伸直。
    |joint4| > thresh 的部分线性惩罚（mean 跨 5 指）；仅贴近 cube 时生效（不干扰接近阶段）。
    治"关节4弯到 1.57 极限的蜷缩爪"（_finger_flex 的 j4 权重 + fingertip 接触奖励共同造成）。
    """
    robot: Articulation = env.scene["robot"]
    ids = []
    for grp in ("thumb", "index", "middle", "ring", "little"):
        j, _ = robot.find_joints(grp + ".*")
        if len(j) > 3:
            ids.append(j[3])          # 第4关节 = 指尖
    ids_t = torch.tensor(ids, device=robot.device)
    abs4 = torch.abs(robot.data.joint_pos[:, ids_t])              # (N,5)
    over = torch.clamp(abs4 - thresh, min=0.0).mean(dim=-1)       # (N,)
    dist = _tcp_cube_dist(env)
    gate = (dist < dist_threshold).float()
    return gate * over


def finger_ratio_reward(
    env: ManagerBasedRLEnv,
    ratio_target: float = 0.6,
    dist_threshold: float = 0.12,
) -> torch.Tensor:
    """(v38) 自然抓取比例奖励：指尖关节4 应 ≤ ratio_target × 中段关节3。
    人手抓取：中段(3)主导弯曲包络、指尖(4)适度伸直；j4 相对 j3 过弯即"蜷缩爪"。
    这是正 shaping（比例内满分 1，超标渐变衰减），非硬惩罚；仅贴近 cube 时生效。
    只约束四指（跳过拇指，对掌解剖不同）。
    """
    robot: Articulation = env.scene["robot"]
    jp = torch.abs(robot.data.joint_pos)                     # (N, num_joints)
    scores = []
    for i, ids in enumerate(_get_finger_joint_ids(robot)):
        if i == 0 or len(ids) <= 3:                          # 跳过拇指
            continue
        j3 = jp[:, ids[2]]                                   # 中段
        j4 = jp[:, ids[3]]                                   # 指尖
        ratio = j4 / (j3 + 1e-6)                             # (N,)
        excess = torch.clamp(ratio - ratio_target, min=0.0)  # 超标量
        scores.append(torch.exp(-excess / 0.5))              # 比例内→1，超标渐变衰减
    raw = torch.stack(scores, dim=-1).mean(dim=-1)           # (N,)
    dist = _tcp_cube_dist(env)
    gate = (dist < dist_threshold).float()
    return gate * raw


def tcp_gated_hand_action_reward(
    env: ManagerBasedRLEnv,
    distance_std: float = 0.1,
    flex_std: float = 0.3,
) -> torch.Tensor:
    """当 TCP 接近物体时，鼓励手指弯曲握紧。
    参考  tcp_gated_hand_action_reward，用 finger_flex 替代力矩。
    用于 Stage 2：TCP 贴掌后才激活手指奖励。
    """
    d = _tcp_cube_dist(env)
    tcp_gate = 1.0 - torch.tanh(d / distance_std)
    robot: Articulation = env.scene["robot"]
    flex = _finger_flex(robot)
    flex_score = torch.tanh(flex / flex_std)
    return tcp_gate * flex_score


# ══════════════════════════════════════════════════════════════
# Sparse Reward: 成功
# ══════════════════════════════════════════════════════════════

def success_reward(
    env: ManagerBasedRLEnv,
    palm_dist_threshold: float = 0.04,  # v20: 0.035→0.04（7cm cube 半边长 3.5cm，需在表面外）
    flex_threshold: float = 0.5,
    lifted_height: float = 0.825,   # Cube 必须被抬离桌面才算真抓起（防"假成功"）
    lin_vel_threshold: float = 0.05,
    ang_vel_threshold: float = 0.10,
) -> torch.Tensor:
    """Stage 2 成功 (0/1): TCP 贴近 + 手指弯曲 + Cube 被抬起（真实抓取）。
    逐步发放（与旧系统一致）：靠近即持续加分，形成价值梯度拉动接近。
    """
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["cube_obj"]
    flex = _finger_flex(robot)
    v_lin = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    v_ang = torch.norm(obj.data.root_ang_vel_w, dim=-1)
    return (
        (_tcp_cube_dist(env) < palm_dist_threshold)
        & (flex > flex_threshold)
        & (obj.data.root_pos_w[:, 2] > lifted_height)
        & (v_lin < lin_vel_threshold) & (v_ang < ang_vel_threshold)
    ).float()


def success_stage1_reward(
    env: ManagerBasedRLEnv,
    palm_dist_threshold: float = 0.06,
    lin_vel_threshold: float = 0.05,
    ang_vel_threshold: float = 0.10,
    d_std: float = 0.02,
    orient_threshold: float | None = None,
) -> torch.Tensor:
    """Stage 1 成功: TCP-Cube 距离 < palm_dist_threshold 给 1.0（饱和），
    palm_dist_threshold ~ palm_dist_threshold+d_std 线性过渡到 0；物体稳定（不要求手指）。
    v87: 二值→饱和型——消除 0/1 硬跳变对价值函数的冲击（success 出现时 value loss 飙升
    是 v6 退化崩溃的放大器）。6cm 内仍满分（保留"达成 6cm"语义），6~8cm 提供够到边界的梯度。
    逐步发放，episode 不终止。

    v93: 加 orient_threshold 平放姿态门控——TCP 姿态角 > orient_threshold 时 success=0。
    reach 只奖距离，倾斜接近是捷径（同样拿 reach，姿态惩罚仅 -0.05 压不住）→ 倾斜 15°
    推 cube（用户 v7 观察）。姿态角成为 success 必要条件后：平放+接近=5.0/步 vs
    倾斜+接近=3.0/步，倾斜净亏 2.0/步 → 策略必然学平放。
    姿态角与 tcp_orientation_penalty 同源（_tcp_init_quat_w 由 reset 事件缓存）。
    """
    obj: RigidObject = env.scene["cube_obj"]
    v_lin = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    v_ang = torch.norm(obj.data.root_ang_vel_w, dim=-1)
    d = _tcp_cube_dist(env)
    close = torch.clamp((palm_dist_threshold + d_std - d) / d_std, min=0.0, max=1.0)
    stable = ((v_lin < lin_vel_threshold) & (v_ang < ang_vel_threshold)).float()
    if orient_threshold is not None:
        robot: Articulation = env.scene["robot"]
        tcp_body_id = _get_body_id(env, "wrist_3_link")
        q_curr = robot.data.body_link_quat_w[:, tcp_body_id]
        if not hasattr(env, "_tcp_init_quat_w"):
            env._tcp_init_quat_w = q_curr.clone()
        q_init = env._tcp_init_quat_w
        dot = (q_curr * q_init).sum(dim=-1).abs().clamp(-1.0, 1.0)
        ang = 2.0 * torch.acos(dot)
        oriented = (ang < orient_threshold).float()
        return close * stable * oriented
    return close * stable


def success_stage3_reward(
    env: ManagerBasedRLEnv,
    palm_dist_threshold: float = 0.15,
    flex_threshold: float = 0.5,
    height_threshold: float = 1.35,
    lin_vel_threshold: float = 0.05,
    ang_vel_threshold: float = 0.10,
) -> torch.Tensor:
    """Stage 3 成功 (0/1): 抓稳 + Cube Z(W) > height_threshold。"""
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["cube_obj"]
    flex = _finger_flex(robot)
    v_lin = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    v_ang = torch.norm(obj.data.root_ang_vel_w, dim=-1)
    return (
        (_tcp_cube_dist(env) < palm_dist_threshold)
        & (flex > flex_threshold)
        & (obj.data.root_pos_w[:, 2] > height_threshold)
        & (v_lin < lin_vel_threshold) & (v_ang < ang_vel_threshold)
    ).float()


# ══════════════════════════════════════════════════════════════
# 惩罚项
# ══════════════════════════════════════════════════════════════

def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """动作变化率惩罚 Σ(a_t − a_{t−1})²，促平滑。
    v89: clamp max=10——288 轮数值爆炸（物理瞬态）时 raw action 大跳，
    平方惩罚可到数百污染 value 训练；封顶后爆炸不拉爆全局。
    """
    diff = env.action_manager.action - env.action_manager.prev_action
    return torch.clamp(torch.sum(torch.square(diff), dim=-1), max=10.0)


def hand_action_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(v77) 惩罚手指关节弯曲量——Stage 1 手指保持伸直。

    play(model_400)：hand 碰 cube 时拇指指根被顶弯向掌心，回升后保持弯——
    旧版只罚动作（action=0 保持当前姿态），手指被碰弯后没有"回伸直"梯度。
    改为罚 max|finger joint_pos|（离伸直位 0 越远越罚），被碰弯后也会回到伸直。
    ⚠️ Stage 2 激活手指奖励时必须 weight→0（否则与抓取弯曲冲突）。

    v91: 改为相对初始姿态——原 max|joint_pos| 把拇指 init 的 ±0.3（故意避开限位）当弯曲
    常数惩罚（每步 -0.15），且诱导策略把拇指弯向 0 撞限位。改为 max(|joint_pos| − |init|)：
    初始弯曲不计入，只有被真正碰弯（超过初始）才罚；比初始更伸直（clamp 到 0）不罚。
    基准用 default_joint_pos（= init_state.joint_pos，单一来源）。
    """
    robot: Articulation = env.scene["robot"]
    ids_list = _get_finger_joint_ids(robot)
    all_ids = torch.cat([t for t in ids_list if t.numel() > 0]).to(robot.device)
    # 初始关节位置（init_state 设定，如拇指 ±0.3 避开限位）——相对它度量弯曲
    # v92: default_joint_pos 兼容 1D (num_joints,) 与 2D (num_envs, num_joints)（部分版本）
    dj = robot.data.default_joint_pos
    init_abs = (dj[0, all_ids] if dj.dim() > 1 else dj[all_ids]).abs()   # (F,)
    bend = torch.abs(robot.data.joint_pos[:, all_ids]) - init_abs   # (N, F) 超出初始的弯曲量（可为负=更伸直）
    val = torch.clamp(bend, min=0.0).max(dim=-1).values             # (N,) 最弯的那根超出的部分
    # v89: clamp max=1.5——手指关节限位 ~±2rad，物理不可能超过；
    #   288 轮出现 max|joint_pos|=20rad（数值爆炸/NaN），封顶防污染。
    return torch.clamp(val, max=1.5)


def joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """关节速度 L2 惩罚（本地 clip 版，覆盖 isaaclab 标准）。
    v89: clamp max=10——288 轮某 env 关节速度爆表（惩罚 -38）污染训练；
    正常速度（<10 rad/s）不受影响，只有物理爆炸被截断。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    l2 = torch.sum(torch.square(robot.data.joint_vel), dim=-1)
    return torch.clamp(l2, max=10.0)


def tcp_velocity_penalty(
    env: ManagerBasedRLEnv,
    vel_std: float = 0.05,
    gate_dist: float | None = None,
) -> torch.Tensor:
    """(W) 惩罚 TCP 移动过快，促稳定接近。vel_std 越小惩罚越重。

    若 gate_dist 非 None（接近减速）：仅在 TCP 距 Cube < gate_dist 时按速度²惩罚，
    远处不罚——避免拖慢远端快速逼近；近端平滑减速，把"直接撞上"变成柔和接触。
    """
    robot: Articulation = env.scene["robot"]
    tcp_body_id = _get_body_id(env, "wrist_3_link")
    # v91: body_vel_w→body_link_vel_w——v85 统一原则：位置/姿态都用 link frame
    # （body_link_pos_w/body_link_quat_w），速度也应取 link frame 原点（body_link_vel_w）。
    # 原 body_vel_w 是关节 frame（body frame 在关节处），对 wrist 偏移 cm 级差异可忽略，
    # 但违反统一性；对齐后与 _get_tcp_pos 的 link frame 语义完全一致。
    v = robot.data.body_link_vel_w[:, tcp_body_id, :3]   # (N, 3) 世界坐标系线速度（link frame）
    speed = torch.norm(v, dim=-1)                     # (N,)
    pen = (speed / vel_std) ** 2                      # (N,) 平方惩罚高速
    if gate_dist is not None:
        d = _tcp_cube_dist(env)
        gate = 1.0 - torch.tanh(d / gate_dist)        # 近→1，远→0 平滑门控
        pen = pen * gate
    return pen


def palm_press_penalty(
    env: ManagerBasedRLEnv,
    clearance: float = 0.005,   # 允许手掌低于 cube 顶面的最小间隙（5mm，轻触级别）
    pen_std: float = 0.015,     # 压入深度归一化（压入 1.5cm → tanh(1)≈0.76）
    xy_gate: float = 0.04,      # 手掌水平投影距 cube 质心的门控（6cm cube 半边长 3cm + 1cm 余量）
) -> torch.Tensor:
    """(v74) 惩罚手掌压穿 cube——治"直接撞上去"。

    用户 2026-08-27 确认：TCP 离手掌 5cm，cube 完全可以在下方（不是 TCP 位置问题）。
    撞的根因：reach/success 只度量 TCP 距离、无方向约束，策略发现"用手掌直接推 cube"
    就能让 TCP 距离变小拿满分，而手掌本体先撞上 cube。
    本项只罚"手掌水平投影在 cube 内 (xy_dist<xy_gate) 且手掌 z 低于 cube 顶面-clearance"，
    压入越深越重（tanh 饱和）。正确悬停（手掌在 cube 上方）不触发，不影响正常接近。
    """
    palm_pos = _get_palm_pos(env)                                     # (N,3)
    cube_pos = _get_cube_pos(env)                                     # (N,3)
    xy_dist = torch.norm(palm_pos[:, :2] - cube_pos[:, :2], dim=-1)   # (N,)
    cube_top = cube_pos[:, 2] + _CUBE_HALF_SIZE                       # (N,) cube 顶面 z
    depth = cube_top - clearance - palm_pos[:, 2]                     # (N,) >0 = 压入量
    press = torch.tanh(torch.clamp(depth, min=0.0) / pen_std)         # (N,)
    over_cube = (xy_dist < xy_gate).float()                           # (N,)
    return over_cube * press


def tcp_too_close_penalty(
    env: ManagerBasedRLEnv,
    min_dist: float = 0.0,
    pen_std: float = 0.02,
    xy_gate: float = 0.06,
) -> torch.Tensor:
    """(v91) 惩罚 TCP 深入 cube 高度带——单调 z 深度版（替代 v74/v86 欧氏 U 形）。

    v74/v86 用"TCP 距质心欧氏距离"（min_dist 5cm）：TCP 穿质心时 d 先减后增（U 形），
    目标态"掌心距顶面 3cm"（TCP 在质心下方 1.5cm，d≈1.5cm）反被罚 1.75/步，
    且穿质心后 d 增大惩罚减轻 → 诱导策略压向 cube（用户观察"撞 cube"根因）。

    新逻辑：只罚 TCP 低于质心（depth = 质心z − TCPz > 0）且水平投影在 cube 内：
    - 掌心距顶面 ≥3cm（TCP 在质心上方）：depth<0 → 不罚（目标态合法）
    - 继续压（TCP 低于质心）：depth 增大 → 线性罚，越深越重（单调无 U 形）
    - TCP 水平远离 cube（xy≥gate）：不罚（侧面接近不误伤）
    """
    tcp = _get_tcp_pos(env)                                       # (N,3)
    cube = _get_cube_pos(env)                                     # (N,3)
    xy_dist = torch.norm(tcp[:, :2] - cube[:, :2], dim=-1)        # (N,)
    depth = cube[:, 2] - tcp[:, 2]                                # (N,) >0 = TCP 低于质心
    pen = torch.clamp(depth - min_dist, min=0.0) / pen_std        # (N,)
    return (xy_dist < xy_gate).float() * pen                      # (N,)


# ══════════════════════════════════════════════════════════════
# TCP 姿态约束（Stage 1+2+3 通用）
# ══════════════════════════════════════════════════════════════

def tcp_orientation_penalty(
    env: ManagerBasedRLEnv,
    ang_std: float = 0.15,
) -> torch.Tensor:
    """惩罚 TCP 偏离 episode 初始姿态。防手腕为够 Cube 而乱扭。
    缓存初始四元数于 env._tcp_init_quat_w，reset 时通过 event 刷新。
    """
    robot: Articulation = env.scene["robot"]
    tcp_body_id = _get_body_id(env, "wrist_3_link")
    q_curr = robot.data.body_link_quat_w[:, tcp_body_id]          # (N, 4) 世界坐标系

    # 缓存初始姿态
    if not hasattr(env, "_tcp_init_quat_w"):
        env._tcp_init_quat_w = q_curr.clone()
    q_init = env._tcp_init_quat_w                             # (N, 4)

    # |q_curr · q_init| — 四元数点积的绝对值
    dot = (q_curr * q_init).sum(dim=-1).abs()                  # (N,)
    dot = torch.clamp(dot, -1.0, 1.0)
    ang = 2.0 * torch.acos(dot)                                # (N,) 0~π

    return (ang / ang_std) ** 2                                  # (N,) 平方惩罚


def cache_tcp_orientation_on_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """reset 回调：缓存当前 TCP 姿态为 episode 初始参考。"""
    robot: Articulation = env.scene["robot"]
    tcp_body_id = _get_body_id(env, "wrist_3_link")
    q = robot.data.body_link_quat_w[:, tcp_body_id]

    if not hasattr(env, "_tcp_init_quat_w") or env._tcp_init_quat_w.shape != q.shape:
        env._tcp_init_quat_w = q.clone()
    else:
        env._tcp_init_quat_w[env_ids] = q[env_ids].clone()

# ══════════════════════════════════════════════════════════════
# 一次性阶段性 bonus
# ══════════════════════════════════════════════════════════════

def tcp_close_bonus_once(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.08,
    bonus_value: float = 5.0,
) -> torch.Tensor:
    """一次性奖励：TCP 首次进入 Cube 的 bonus_range 内时发放，
    之后不再重复。需要用 reset 事件初始化 _tcp_close_granted mask。
    """
    # 懒初始化 mask
    N = env.num_envs
    if not hasattr(env, "_tcp_close_granted") or env._tcp_close_granted.shape[0] != N:
        env._tcp_close_granted = torch.zeros(N, dtype=torch.bool, device=env.device)

    dist = _tcp_cube_dist(env)                                   # (N,)
    close = dist < distance_threshold                               # (N,)
    first_time = close & ~env._tcp_close_granted                    # (N,)
    reward = first_time.float() * bonus_value                       # (N,)
    env._tcp_close_granted = env._tcp_close_granted | close         # 锁定
    return reward


def reset_tcp_close_granted(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """reset 回调：清除指定 envs 的 tcp_close 一次性奖励标记。"""
    N = env.num_envs
    if not hasattr(env, "_tcp_close_granted") or env._tcp_close_granted.shape[0] != N:
        env._tcp_close_granted = torch.zeros(N, dtype=torch.bool, device=env.device)
    env._tcp_close_granted[env_ids] = False


def object_distance_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.25,
) -> torch.Tensor:
    """(W) TCP-Cube > threshold 时线性惩罚，防推飞/掉落。"""
    return torch.clamp(_tcp_cube_dist(env) - threshold, min=0.0)


# ══════════════════════════════════════════════════════════════
# 间接接触检测（无触觉 API 的替代方案）
# ══════════════════════════════════════════════════════════════

def grasp_contact_reward(
    env: ManagerBasedRLEnv,
    flex_threshold: float = 0.5,
    displacement_threshold: float = 0.003,  # 3mm 位移 = 碰到了
    bonus_per_step: float = 2.0,
) -> torch.Tensor:
    """间接接触检测：手指弯曲 + Cube 被扰动 = 真抓到了。
    
    无需 ContactSensor，利用物理事实——手指碰到 Cube 必然导致微小位移。
    reset 事件中由 cache_object_init_pos_on_reset 初始化 _obj_init_pos_w。
    """
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["cube_obj"]

    # 手指是否在弯曲
    flex = _finger_flex(robot)                                       # (N,)
    bending = flex > flex_threshold                                   # (N,)

    # Cube 是否被扰动（脱离初始静止位置）
    if not hasattr(env, "_obj_init_pos_w"):
        env._obj_init_pos_w = obj.data.root_pos_w.clone()
    displacement = torch.norm(obj.data.root_pos_w - env._obj_init_pos_w, dim=-1)  # (N,)
    perturbed = displacement > displacement_threshold                 # (N,)

    return bending.float() * perturbed.float() * bonus_per_step       # (N,)


# ══════════════════════════════════════════════════════════════
# 指尖触觉（Stage 2）
# ══════════════════════════════════════════════════════════════

def contact_force_reward(
    env: ManagerBasedRLEnv,
    force_threshold: float = 0.1,
    deadzone: float = 0.05,          # 参考 ：去噪，< 0.05N 的轻抚不算接触
    bonus_per_contact: float = 1.0,
) -> torch.Tensor:
    """二值接触检测：指尖力 > force_threshold N → 1 分/指。
    参考  binary 模式 + deadzone——碰到就奖，忽略微小接触噪声。
    """
    from .observations import fingertip_contact_force
    forces = fingertip_contact_force(env)                     # (N, 5)
    forces = torch.clamp(forces - deadzone, min=0.0)           # deadzone 去噪
    contact_mask = (forces > force_threshold).float()           # (N, 5) 0/1
    return contact_mask.sum(dim=-1) * bonus_per_contact          # (N,) 0~5


def grip_force_reward(
    env: ManagerBasedRLEnv,
    force_std: float = 1.0,
) -> torch.Tensor:
    """(v50) 连续握力奖励：力越大分越高（非二值）。

    二值 contact_force 的缺陷（v49 诊断暴露）：>0.02N 即满分，没有"压多紧"的梯度——
    策略让 5 指都轻轻擦到（各 1 分）就满分，无动力加压。诊断实测接触力仅 0.03~0.09N。
    本项每指 1−exp(−f/σ) 连续 0→1：
      0.1N→0.10  0.5N→0.39  1N→0.63  2N→0.86
    从"擦到"到"握紧"(2N≈握起 0.2kg) 全程有梯度，策略才会持续加压。
    """
    from .observations import fingertip_contact_force
    forces = fingertip_contact_force(env)                     # (N, 5) 单位 N
    score = 1.0 - torch.exp(-forces / force_std)              # (N, 5) 0→1 连续
    return score.sum(dim=-1)                                  # (N,) 0~5


def _fingertip_force_vectors(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 五个指尖接触力向量 → (N,5,3)，单位 N。force_matrix_w[:,0,0,:] = 世界系力向量。"""
    vecs = torch.zeros(env.num_envs, 5, 3, device=env.device)
    sensor_names = ["contact_thumb", "contact_index", "contact_middle", "contact_ring", "contact_little"]
    for i, name in enumerate(sensor_names):
        sensor = env.scene.sensors[name]
        fmat = sensor.data.force_matrix_w                     # (N, B, M, 3)
        if fmat is not None and fmat.numel() > 0:
            vecs[:, i] = fmat[:, 0, 0, :]
    return vecs


def opposition_reward(
    env: ManagerBasedRLEnv,
    gate_dist: float = 0.12,
    scale: float = 1.0,
) -> torch.Tensor:
    """(v52) 对向夹持奖励（力封闭的平滑形式）——专治"四指挤同侧、无法夹"。

    背景（v51 诊断）：四指全压 +x 面 → 所有力同向 → 无对向 → cube 侧滑。
    grip_force（总力）会被单侧猛压骗到高分，无法区分"抓得稳"与"单侧推"。

    本项：把每指接触力向量按水平方向分别聚合到 4 个面，奖励**相对两面的力取 min**：
        R = min(F_+x, F_-x) + min(F_+y, F_-y)，再 tanh 饱和到 0~1
    - 单侧压：一侧力大、对侧 0 → min=0 → 得 0 分 ✅ 正确惩罚
    - 对向夹持：两侧都有力 → min>0 → 有分 ✅ 这才是能夹住的形式
    对力符号不敏感（relu 分方向后 min 对称），平滑（力连续），带 TCP 距离门控。
    """
    forces = _fingertip_force_vectors(env)                    # (N,5,3)
    fx = forces[..., 0]                                       # (N,5) 水平 x 分量
    fy = forces[..., 1]                                       # (N,5) 水平 y 分量
    F_px = torch.relu(fx).sum(dim=-1)                          # 朝 +x 的合力
    F_nx = torch.relu(-fx).sum(dim=-1)                         # 朝 -x 的合力
    F_py = torch.relu(fy).sum(dim=-1)
    F_ny = torch.relu(-fy).sum(dim=-1)
    opp = torch.minimum(F_px, F_nx) + torch.minimum(F_py, F_ny)  # (N,)
    opp = torch.tanh(opp / scale)                              # 0→1 平滑饱和
    dist = _tcp_cube_dist(env)
    gate = (dist < gate_dist).float()
    return gate * opp


def excess_force_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 5.0,
) -> torch.Tensor:
    """(W) 指尖力 > threshold N 惩罚，防捏碎。"""
    from .observations import fingertip_contact_force
    forces = fingertip_contact_force(env)                     # (N, 5)
    return torch.clamp(forces - threshold, min=0.0).sum(dim=-1)


# ══════════════════════════════════════════════════════════════
# 举升（Stage 3）
# ══════════════════════════════════════════════════════════════

def lift_reward(
    env: ManagerBasedRLEnv,
    gate_dist: float = 0.3,
    gate_steep: float = 10.0,
    flex_thresh: float = 0.3,
    flex_steep: float = 5.0,
    z_thresh: float = 1.25,
) -> torch.Tensor:
    """(W) Cube Z > z_thresh 时奖励举升。门控: 必须贴掌+手指弯曲才给分。"""
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["cube_obj"]
    close = torch.sigmoid(gate_steep * (gate_dist - _tcp_cube_dist(env)))
    flexed = torch.sigmoid(flex_steep * (_finger_flex(robot) - flex_thresh))
    lift = torch.clamp(obj.data.root_pos_w[:, 2] - z_thresh, min=0.0)
    return close * flexed * torch.sqrt(lift)


# ══════════════════════════════════════════════════════════════
# 物体抬升（参考  object_is_lifted）
# ══════════════════════════════════════════════════════════════

def object_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.82,
) -> torch.Tensor:
    """Cube Z > minimal_height → 1.0，否则 0。
    参考 ——抬升 1mm 就意味着真抓到了，是最可靠的抓取信号。
    """
    obj: RigidObject = env.scene["cube_obj"]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def palm_height_reward(
    env: ManagerBasedRLEnv,
    z_thresh: float = 1.30,
) -> torch.Tensor:
    """(W) 手掌 Z > z_thresh 时奖励手臂抬高。不依赖抓握。"""
    return torch.clamp(_get_palm_pos(env)[:, 2] - z_thresh, min=0.0)
