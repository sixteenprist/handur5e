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

# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════

# 手指关节按名称模式匹配（兼容新 USD 命名，如 thumb4_hoint / 关节顺序变化）
_FINGER_JOINT_PATTERNS = {
    "thumb": "thumb.*",
    "index": "index.*",
    "middle": "middle.*",
    "ring": "ring.*",
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


_FINGERTIP_NAMES = ["thumb4", "index4", "middle4", "ring4", "little4"]
_CUBE_HALF_SIZE = 0.035  # 7cm Cube 半边长 (m)（v20: 6cm→7cm）
_BODY_OFFSET = torch.tensor([0.03, -0.02, 0.06])  # TCP = base_link_1 + offset (B)，与 env_cfg 控制器一致


# ══════════════════════════════════════════════════════════════
# 基础工具（世界坐标系，供本模块内部使用）
# ══════════════════════════════════════════════════════════════

def _get_palm_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 手掌 base_link_1 位置 → (N,3)."""
    robot: Articulation = env.scene["robot"]
    return robot.data.body_link_pos_w[:, int(robot.find_bodies("base_link_1")[0][0])]


def _get_cube_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) Cube 质心位置 → (N,3)."""
    return env.scene["cube_obj"].data.root_pos_w


def _get_tcp_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) TCP 位置 = palm_pos(W) + R_palm * body_offset(B)."""
    robot: Articulation = env.scene["robot"]
    palm_id = int(robot.find_bodies("base_link_1")[0][0])
    palm_pos = robot.data.body_link_pos_w[:, palm_id]
    palm_quat = robot.data.body_link_quat_w[:, palm_id]

    from isaaclab.utils.math import quat_apply
    offset = _BODY_OFFSET.to(palm_pos.device).unsqueeze(0).expand(palm_pos.shape[0], -1)
    return palm_pos + quat_apply(palm_quat, offset)


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
    return robot.data.body_link_pos_w[:, int(robot.find_bodies(name)[0][0])]


def _fingertip_cube_surface_dists(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 五指尖→Cube 表面距离 → (N,5)。扣除了 Cube 半边长。每指独立。"""
    cube = _get_cube_pos(env)
    dists = []
    for t in _FINGERTIP_NAMES:
        d = torch.norm(_get_fingertip_pos(env, t) - cube, dim=-1)
        d = torch.clamp(d - _CUBE_HALF_SIZE, min=0.0)
        dists.append(d.unsqueeze(-1))
    return torch.cat(dists, dim=-1)


# 每指 4 关节权重（v22：拇指与其他四指不同——用户确认 拇指1=对掌旋转、2=侧摆）
# 四指 [侧摆1, 掌根2, 中段3, 指尖4] = [0.2, 0.8, 1.5, 0.8]（重点压中段3）
# 拇指 [对掌1, 侧摆2, 中段3, 指尖4] = [1.2, 0.2, 1.5, 0.8]（对掌1 保留高权重，它是拇指抓握关键）
# v18 崩溃根因 = 只有掌根(2)弯、中段(3)不弯 → 指尖够不到、只有压力没接触 → 挤飞
_FINGER_JOINT_WEIGHTS = [
    torch.tensor([1.2, 0.2, 1.5, 0.8]),  # 拇指: 对掌1 / 侧摆2 / 中段3 / 指尖4
    torch.tensor([0.2, 0.8, 1.5, 0.8]),  # 食指: 侧摆1 / 掌根2 / 中段3 / 指尖4
    torch.tensor([0.2, 0.8, 1.5, 0.8]),  # 中指
    torch.tensor([0.2, 0.8, 1.5, 0.8]),  # 无名指
    torch.tensor([0.2, 0.8, 1.5, 0.8]),  # 小指
]


def _finger_flex(robot: Articulation) -> torch.Tensor:
    """手指弯曲度量 (rad)。0=伸直, ~1.57=全弯。
    关节加权（v21/v22）：每指 4 关节，中段3 权重最高（45%），迫使策略弯中段而非只弯掌根；
    拇指对掌1 权重 1.2（对掌是拇指抓握关键，不能降权）。
    跨指 0.7×最不弯 + 0.3×平均，防拇指独自弯完。
    """
    abs_pos = torch.abs(robot.data.joint_pos)
    per_finger = []
    for i, ids in enumerate(_get_finger_joint_ids(robot)):
        w = _FINGER_JOINT_WEIGHTS[i].to(abs_pos.device)  # (4,)
        per_finger.append((abs_pos[:, ids] * w).sum(dim=1) / w.sum())
    per_finger = torch.stack(per_finger, dim=0)  # (5, N) 加权平均
    return 0.7 * per_finger.min(dim=0).values + 0.3 * per_finger.mean(dim=0)


# ══════════════════════════════════════════════════════════════
# Dense Reward: 接近与抓取
# ══════════════════════════════════════════════════════════════

def reach_reward(
        env: ManagerBasedRLEnv,
        std: float = 0.25,
) -> torch.Tensor:
    """(W) TCP→Cube 距离奖励 1−tanh(d/σ)。
    各处梯度均匀，不像高斯在远处几乎平直。
    0.5m→0.04, 0.3m→0.17, 0.2m→0.34, 0.1m→0.62, 0.04m→0.84。
    """
    d = _tcp_cube_dist(env)
    return 1.0 - torch.tanh(d / std)


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
    dist = _tcp_cube_dist(env)  # (N,)
    gate = (dist < dist_threshold).float()  # 硬门：0 或 1
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
            ids.append(j[2])  # 第3个关节 = 中段
    ids_t = torch.tensor(ids, device=robot.device)
    abs3 = torch.abs(robot.data.joint_pos[:, ids_t])  # (N,4)
    flex3 = 0.5 * abs3.mean(dim=1) + 0.5 * abs3.min(dim=1).values  # (N,)
    dist = _tcp_cube_dist(env)
    gate = (dist < dist_threshold).float()
    return gate * torch.min(flex3, torch.tensor(max_flex, device=flex3.device))


def fingertip_contact_reward(
        env: ManagerBasedRLEnv,
        std: float = 0.04,
        gate_std: float = 0.15,
) -> torch.Tensor:
    """(W) 指尖→Cube 表面距离奖励 × TCP 门控。
    每指独立 1−tanh(d/σ)，v18 聚合 = 0.5·mean + 0.5·min：
    - mean 半：平滑梯度，每根手指独立可学（v16 解决 min 难学卡死）
    - min 半：专门压"最远那根"（无名指/小指），防 mean 稀释导致它们不弯
    v23: std 0.08→0.04——0.08 时差 1-2cm 就 82%，无"最后 1cm"梯度；
    0.04 只奖励真正贴到，恢复接触梯度（策略已贴近，不再怕尖梯度难学）。
    """
    dists = _fingertip_cube_surface_dists(env)  # (N,5)
    per_finger = 1.0 - torch.tanh(dists / std)  # (N,5)
    raw = 0.5 * per_finger.mean(dim=-1) + 0.5 * per_finger.min(dim=-1).values  # (N,)
    # Soft TCP gate
    tcp_dist = _tcp_cube_dist(env)
    gate = 1.0 - torch.tanh(tcp_dist / gate_std)
    return raw * gate


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
        lifted_height: float = 0.825,  # Cube 必须被抬离桌面才算真抓起（防"假成功"）
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
        palm_dist_threshold: float = 0.04,
        lin_vel_threshold: float = 0.05,
        ang_vel_threshold: float = 0.10,
) -> torch.Tensor:
    """Stage 1 成功 (0/1): TCP-Cube 距离 < palm_dist_threshold, 物体稳定（不要求手指）。
    逐步发放（与旧系统一致），episode 不终止，留步给后续抓取/举升阶段。
    """
    obj: RigidObject = env.scene["cube_obj"]
    v_lin = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    v_ang = torch.norm(obj.data.root_ang_vel_w, dim=-1)
    return (
            (_tcp_cube_dist(env) < palm_dist_threshold)
            & (v_lin < lin_vel_threshold) & (v_ang < ang_vel_threshold)
    ).float()


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
    """动作变化率惩罚 Σ(a_t − a_{t−1})²，促平滑。"""
    return torch.sum(torch.square(
        env.action_manager.action - env.action_manager.prev_action), dim=-1)


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
    palm_id = int(robot.find_bodies("base_link_1")[0][0])
    v = robot.data.body_vel_w[:, palm_id, :3]  # (N, 3) 世界坐标系线速度
    speed = torch.norm(v, dim=-1)  # (N,)
    pen = (speed / vel_std) ** 2  # (N,) 平方惩罚高速
    if gate_dist is not None:
        d = _tcp_cube_dist(env)
        gate = 1.0 - torch.tanh(d / gate_dist)  # 近→1，远→0 平滑门控
        pen = pen * gate
    return pen


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
    palm_id = int(robot.find_bodies("base_link_1")[0][0])
    q_curr = robot.data.body_link_quat_w[:, palm_id]  # (N, 4) 世界坐标系

    # 缓存初始姿态
    if not hasattr(env, "_tcp_init_quat_w"):
        env._tcp_init_quat_w = q_curr.clone()
    q_init = env._tcp_init_quat_w  # (N, 4)

    # |q_curr · q_init| — 四元数点积的绝对值
    dot = (q_curr * q_init).sum(dim=-1).abs()  # (N,)
    dot = torch.clamp(dot, -1.0, 1.0)
    ang = 2.0 * torch.acos(dot)  # (N,) 0~π

    return (ang / ang_std) ** 2  # (N,) 平方惩罚


def cache_tcp_orientation_on_reset(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
):
    """reset 回调：缓存当前 TCP 姿态为 episode 初始参考。"""
    robot: Articulation = env.scene["robot"]
    palm_id = int(robot.find_bodies("base_link_1")[0][0])
    q = robot.data.body_link_quat_w[:, palm_id]

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

    dist = _tcp_cube_dist(env)  # (N,)
    close = dist < distance_threshold  # (N,)
    first_time = close & ~env._tcp_close_granted  # (N,)
    reward = first_time.float() * bonus_value  # (N,)
    env._tcp_close_granted = env._tcp_close_granted | close  # 锁定
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
    flex = _finger_flex(robot)  # (N,)
    bending = flex > flex_threshold  # (N,)

    # Cube 是否被扰动（脱离初始静止位置）
    if not hasattr(env, "_obj_init_pos_w"):
        env._obj_init_pos_w = obj.data.root_pos_w.clone()
    displacement = torch.norm(obj.data.root_pos_w - env._obj_init_pos_w, dim=-1)  # (N,)
    perturbed = displacement > displacement_threshold  # (N,)

    return bending.float() * perturbed.float() * bonus_per_step  # (N,)


# ══════════════════════════════════════════════════════════════
# 指尖触觉（Stage 2）
# ══════════════════════════════════════════════════════════════

def contact_force_reward(
        env: ManagerBasedRLEnv,
        force_threshold: float = 0.1,
        deadzone: float = 0.05,  # 参考 ：去噪，< 0.05N 的轻抚不算接触
        bonus_per_contact: float = 1.0,
) -> torch.Tensor:
    """二值接触检测：指尖力 > force_threshold N → 1 分/指。
    参考  binary 模式 + deadzone——碰到就奖，忽略微小接触噪声。
    """
    from .observations import fingertip_contact_force
    forces = fingertip_contact_force(env)  # (N, 5)
    forces = torch.clamp(forces - deadzone, min=0.0)  # deadzone 去噪
    contact_mask = (forces > force_threshold).float()  # (N, 5) 0/1
    return contact_mask.sum(dim=-1) * bonus_per_contact  # (N,) 0~5


def excess_force_penalty(
        env: ManagerBasedRLEnv,
        threshold: float = 5.0,
) -> torch.Tensor:
    """(W) 指尖力 > threshold N 惩罚，防捏碎。"""
    from .observations import fingertip_contact_force
    forces = fingertip_contact_force(env)  # (N, 5)
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
