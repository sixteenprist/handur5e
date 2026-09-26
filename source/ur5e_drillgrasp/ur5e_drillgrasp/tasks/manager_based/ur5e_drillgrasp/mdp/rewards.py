# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MDP 奖励函数。

三阶段课程学习对应关系（[2026-09-14] 名称同步为现役函数）：
  Stage 1 接近 ─── reach + success_stage1（+ tcp_orientation / action_rate 惩罚）
  Stage 2 抓取 ─── + finger_reaching → finger_contact → grip（力封闭）→ contact_hold
  Stage 3 举升 ─── + object_goal_tracking + object_goal_bonus

[2026-09-14 四指共享适配] 凡"每指量"的聚合，一律按"拇指组（独立 4D）+ 四指组（共享 4D）"两级结构
（见 _thumb_four_split）：组内 mean（吸收指尖几何残差），组间 0.5/0.5 或 min（与 grip 的两侧语义同构）。
禁用 5 指等权 mean/min——四指共享下 mean 有 4:1 梯度偏置、min 被"最难贴那根"钉死。

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
from .observations import (
    _BODY_OFFSET, _CUBE_HALF_SIZE, _FINGERTIP_NAMES, _get_body_id,
    ACTIVE_TIP_NAMES, ACTIVE_FOUR_INDICES,
)


# ══════════════════════════════════════════════════════════════
# 基础工具（世界坐标系，供本模块内部使用）
# ══════════════════════════════════════════════════════════════

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


def _get_fingertip_pos(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """(W) 单个指尖 link 位置 → (N,3)."""
    robot: Articulation = env.scene["robot"]
    return robot.data.body_link_pos_w[:, _get_body_id(env, name)]


def _fingertip_cube_surface_dists(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 五指尖→Cube 表面距离 → (N,5)。扣除了 Cube 半边长。每指独立。

    [2026-09-07] 曾改精确边界框距离修小指/无名指颤动，但精确距离让小指/无名指“贴棱边即满分”→
      finger_reaching 在棱边处梯度消失→手指对应通道无梯度→熵漂移快速崩。
      暂回退球近似（|指尖-质心|-半边长）：颤动问题后置，先专注逼力主线。
    """
    cube = _get_cube_pos(env)
    dists = []
    for t in _FINGERTIP_NAMES:
        d = torch.norm(_get_fingertip_pos(env, t) - cube, dim=-1)
        d = torch.clamp(d - _CUBE_HALF_SIZE, min=0.0)
        dists.append(d.unsqueeze(-1))
    return torch.cat(dists, dim=-1)


def _tip_cube_box_dist(env: ManagerBasedRLEnv, tip_name: str) -> torch.Tensor:
    """(N,) 指尖→Cube 的**精确盒面距离**（AABB）：‖clamp(|p−c|−h, min=0)‖。

    [v16 诊断修复] 旧度量用球近似（‖p−c‖−半边长）。指尖卷到顶面**棱边**附近时球近似
      严重高估：网格实测真实盒面距 1.34cm 时球近似报 ~3.5cm → surface 奖励在最后
      1cm 对"继续卷"反而降分（越卷球距越大）→ 策略停在 3.5cm 平台且 m4 不卷。
      盒面距离在面/棱/角处都精确，与接触物理一致（contact_offset=5mm）。
    """
    p_tip = _get_fingertip_pos(env, tip_name)
    c = _get_cube_pos(env)
    dvec = (p_tip - c).abs() - _CUBE_HALF_SIZE
    return torch.norm(dvec.clamp(min=0.0), dim=-1)


def _fingertip_cube_center_dists(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 五指尖→Cube 质心的纯欧氏距离 → (N,5)。**不扣**半边长（SoftHand 对齐版，finger_reaching 专用）。

    [2026-09-15 对齐调查] 对照 SoftHandGrasping 项目（抓的同样是 Isaac 自带 DexCube）：
      `fingertips_object_distance_reward` 用"指尖(link +1.5cm 表面偏移)→质心"纯距离 + std=0.10 宽核，
      且其"clamp(d-0.02) 近似物体表面"一行是**注释掉的**——即 SoftHand 不做精确表面测量，
      形状误差（cube 棱角 vs 球近似）由宽核吸收，也不存在"clamp 提前饱和区"。
    旧实现（_fingertip_cube_surface_dists：扣 3cm 球近似 + 窄核 std=0.03）实测两缺陷：
      ① 棱角区提前饱和：离棱 1~2cm 悬空即 d≤0 → 满分 1.0（球近似误差全暴露在窄核上）→ 无动力钻入；
      ② 中远场死区：表面 5cm 外 ≈0.003（1-tanh(5/3)）→"伸手"起步几乎无梯度（与"手指接近学不出"吻合）。
    """
    cube = _get_cube_pos(env)
    dists = []
    for t in _FINGERTIP_NAMES:
        d = torch.norm(_get_fingertip_pos(env, t) - cube, dim=-1)
        dists.append(d.unsqueeze(-1))
    return torch.cat(dists, dim=-1)


def _thumb_four_split(per: torch.Tensor, four_idx: tuple | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """每指量 (N,5) → (拇指组 (N,), 工作对组 (N,))。手指顺序 = _FINGERTIP_NAMES。

    [2026-09-21 v7 用户要求·坑修复] 工作对组由 `[2:4]` 均值（中指+无名指）改为**仅中指 [2]**：
      资产里无名指碰撞已关闭（力/接触信号恒零），均值会把中指量系统性稀释一半
      （力项低估 2×、占空比与"拇指入场"诊断失真）；且任务约定就是"拇指+中指两指捏取、
      其余三指不参与"。影响所有走本函数的项（finger_contact / contact_hold /
      finger_reaching / finger_close-prox / opposition 系列）。
    [历史 2026-09-16] 当时从 [1:5] 改 [2:4] 的原因（消除食/小指两个零的稀释）同样适用于
      本次——现在连无名指也关了，故收敛到仅中指。
    """
    if four_idx is None:
        four_idx = ACTIVE_FOUR_INDICES
    idx = list(four_idx)
    return per[:, 0], per[:, idx].mean(dim=-1)

# 每指 4 关节权重 [侧摆1, 掌根2, 中段3, 指尖4]（跨指聚合 0.7·min+0.3·mean 见 _finger_flex）
_FINGER_JOINT_WEIGHTS = [
    # [v19c 用户观察] 拇指 j1 0.8→0.0：|侧摆| 被当"弯曲"奖励 → 策略把拇指基座越拧越过、
    #   指尖扫出侧面范围（用户 play 观察 + 探针 t1=-1.16/t2=-1.57/t3=t4=0 证实）。
    #   回退：0.8。
    torch.tensor([0.0, 0.5, 1.5, 1.0]),  # 拇指
    torch.tensor([0.1, 1.2, 0.5, 0.0]),  # 食指
    torch.tensor([0.1, 1.2, 0.5, 0.0]),  # 中指
    torch.tensor([0.1, 1.2, 0.5, 0.0]),  # 无名指
    torch.tensor([0.1, 1.2, 0.5, 0.0]),  # 小指
]


def _finger_flex_per_finger(robot: Articulation) -> torch.Tensor:
    """每根手指的加权弯曲度量 → (5, N)。0=伸直, ~1.57=全弯。
    （从 _finger_flex 抽出，供手指一致性奖励等复用）
    """
    abs_pos = torch.abs(robot.data.joint_pos)
    per_finger = []
    for i, ids in enumerate(_get_finger_joint_ids(robot)):
        w = _FINGER_JOINT_WEIGHTS[i].to(abs_pos.device)      # (4,)
        q = abs_pos[:, ids]                                   # (N,4) [j1,j2,j3,j4]
        if i > 0 and len(ids) >= 4:
            # 层级门控：[v4.2] 0.60→0.25/0.35：原阈值要求 j2 弯到 0.6 才让 j3 计分——
            #   j2 部分弯曲（0.2~0.5）时 j3 白弯、弯曲收益断链 → 策略选择"伸直不动"。
            #   现在 j2>0.25 即开 j3、j3>0.35 即开 j4，弯曲收益链连续。
            g3 = torch.sigmoid(8.0 * (q[:, 1] - 0.25))
            g4 = torch.sigmoid(8.0 * (q[:, 2] - 0.35))
            contrib = w[0]*q[:, 0] + w[1]*q[:, 1] + w[2]*q[:, 2]*g3 + w[3]*q[:, 3]*g4
            per_finger.append(contrib / w.sum())
        else:
            per_finger.append((q * w).sum(dim=1) / w.sum())
    return torch.stack(per_finger, dim=0)                    # (5, N)


def _finger_flex(robot: Articulation) -> torch.Tensor:
    """手指弯曲度量 (rad)。0=伸直, ~1.57=全弯。
    关节加权 + 层级门控（v41）：
    - 四指 掌根2 权重 1.2——根节先弯、弯得多
    - 后续关节在上一关节已弯（>0.15）的基础上才计分：j3 计分需 j2 弯，j4 计分需 j3 弯
      （sigmoid 软门）——自然的手指顺序弯曲，防"指尖独弯的蜷缩爪"
    - 拇指保持原权重，不做层级门控（解剖不同）
    跨指 0.7×最不弯 + 0.3×平均，防拇指独自弯完；[2026-09-14] 跨指聚合改为"拇指组 vs 四指组"两组。
    """
    per = _finger_flex_per_finger(robot)                     # (5, N)
    # [2026-09-14 四指共享适配] 跨指 0.7×min + 0.3×mean（5 指）→"拇指组 vs 四指组"两组：
    #   四指同通路驱动应视为整体（组内 mean 吸收负载差异）；组间 min 保持"拇指也要弯"约束。
    _fi = list(ACTIVE_FOUR_INDICES)
    thumb, four = per[0], per[_fi].mean(dim=0)               # 各 (N,)（四指组=ACTIVE_FOUR_INDICES）
    return 0.7 * torch.minimum(thumb, four) + 0.3 * (0.5 * thumb + 0.5 * four)


def fingertip_press_penalty(
    env: ManagerBasedRLEnv,
    force_std: float = 1.0,
    deadzone: float = 0.02,
    gate_dist: float = 0.02,
) -> torch.Tensor:
    """(SoftHand 借鉴) 指尖合力向下的分量惩罚——防手指把 cube 压向桌面。

    若策略"下压"cube（无对向夹持），指尖合力 Z 向下分量大 → 罚；
    水平对向夹持（opposition 需要的力封闭）则 Z 分量小 → 少罚。
    世界系 Z 判断（重力方向向下），带 TCP 距离门控。

    [2026-09-03] 线性 down/force_std + clamp max=10 → tanh 软饱和：
      旧版物理瞬态 down 到 10N+ → clamp 10 → 单步惩罚 -5（weight -0.5 时），比所有正奖励大
      → value loss 尖峰 → policy 崩塌（黄线 Step 1800~2200 崩溃的触发源）。
      tanh 软饱和有界：down=1N→0.76、2N→0.96、10N→1.0，单步惩罚恒 ≤ weight，
      正常抓取（1~2N）与旧版接近，物理瞬态不再灌大惩罚。
    """
    forces = _fingertip_force_vectors(env)                   # (N,5,3) 世界系
    down = torch.clamp(-forces[..., 2].sum(dim=-1), min=0.0)  # (N,) 向下合力
    if deadzone > 0.0:
        down = torch.clamp(down - deadzone, min=0.0)
    press = torch.tanh(down / force_std)                      # (N,) 0~1 软饱和，无尖峰
    d = _tcp_cube_dist(env)
    gate = (d < gate_dist).float()
    return gate * press


# ══════════════════════════════════════════════════════════════
# Dense Reward: 接近与抓取
# ══════════════════════════════════════════════════════════════

def reach_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    sat_dist: float | None = None,
    sat_transition: float = 0.01,
    orient_threshold: float | None = None,
    z_std: float | None = None,
) -> torch.Tensor:
    """(W) TCP→Cube 距离奖励 1−tanh(d/σ)。
    各处梯度均匀，不像高斯在远处几乎平直。
    0.5m→0.04, 0.3m→0.17, 0.2m→0.34, 0.1m→0.62, 0.04m→0.84。

    ═══ 最初版本（激活）：纯 tanh，无饱和、无姿态门控 ═══
    [用户] z_std: TCP 只允许在 cube 质心及以上（与 success 一致）——
      质心以上满分，质心下方 z_std 内线性衰减到 0（禁从下方接近）。
    切换回当前版本（饱和 + 姿态门控）时：启用下方注释段，并注释掉上面的 return。

    ═══ 当前版本（v82/v91/v93，注释保留）═══
    v82: sat_dist 非 None 时，d < sat_dist 给满分 1.0（饱和）——消除"越近越好"。
    v91: sat_transition 平滑过渡（[sat, sat+trans] 内 1→基础值线性）。
    v93: orient_threshold 姿态门控——TCP 姿态角 > 阈值时 reach 归零（治倾斜/翻转捷径）。
    """
    d = _tcp_cube_dist(env)
    r = 1.0 - torch.tanh(d / std)
    if z_std is not None:
        # [用户] 方向约束：TCP z >= cube 质心 z → 满分；质心下方 z_std 内线性衰减 → 0
        tcp_z = _get_tcp_pos(env)[:, 2]
        cube_z = _get_cube_pos(env)[:, 2]
        above = torch.clamp(1.0 + (tcp_z - cube_z) / z_std, min=0.0, max=1.0)
        r = r * above
    return r

    # ---- 当前版本（切换时启用本段，并注释掉上面的 return）----
    # d = _tcp_cube_dist(env)
    # r = 1.0 - torch.tanh(d / std)
    # if sat_dist is not None:
    #     if sat_transition > 0:
    #         d_sat = sat_dist + sat_transition
    #         r_end = 1.0 - torch.tanh(torch.tensor(d_sat / std, device=d.device))
    #         ramp = torch.clamp((d_sat - d) / sat_transition, min=0.0, max=1.0)
    #         r_sat = r_end + (1.0 - r_end) * ramp
    #         r = torch.where(d < d_sat, r_sat, r)
    #     else:
    #         r = torch.where(d < sat_dist, torch.ones_like(r), r)
    # if orient_threshold is not None:
    #     robot: Articulation = env.scene["robot"]
    #     tcp_body_id = _get_body_id(env, "wrist_3_link")
    #     q_curr = robot.data.body_link_quat_w[:, tcp_body_id]
    #     if not hasattr(env, "_tcp_init_quat_w"):
    #         env._tcp_init_quat_w = q_curr.clone()
    #     q_init = env._tcp_init_quat_w
    #     dot = (q_curr * q_init).sum(dim=-1).abs().clamp(-1.0, 1.0)
    #     ang = 2.0 * torch.acos(dot)
    #     oriented = (ang < orient_threshold).float()
    #     r = r * oriented
    # return r


def finger_reaching_reward(
    env: ManagerBasedRLEnv,
    touch_std: float = 0.10,
    dist_threshold: float = 0.15,
    d_std: float = 0.05,
) -> torch.Tensor:
    """(2026-09-04) 指尖接近 cube 面奖励——密集梯度（bootstrap 用）。

    与 finger_contact（纯力判据）分离：本项只奖励"指尖靠近 cube 面"，
    宽过渡带（touch_std=3cm）提供从远到近的连续梯度，避免稀疏。
    不奖励"贴面有力"——那由 finger_contact 的纯力 + 滞回判据负责。
    [拆分原因] 原 touch×force 混合结构：touch 几何距离在"接近"即饱和 → 白拿底分，
      而 force 在"接触"才激活 → "接近→接触"梯度断档。拆成"距离引导 + 力判据"两段。
    [2026-09-16 历史] S2-α4 曾试"近端精核"（本核叠加近端陡核）——改现有项的值分布
      → value 失配 → 迭代 1419 崩，已整体回退。替代路线：启用 finger_close（现成
      "弯曲引导"项）等"加零项"安全方式（见 env_cfg）。
    """
    # [2026-09-15 SoftHand 对齐] 纯质心距离（不扣半径）+ 宽核（touch_std 0.03→0.10 由 env_cfg 传入）：
    #   SoftHand 抓同样的方块时用"质心距离 + std=0.10"，不做精确表面测量——形状误差由宽核吸收。
    #   旧"扣 3cm 球近似 + 窄核"两缺陷：①棱角悬空 1~2cm 提前满分（无动力钻入）；
    #   ②表面 5cm 外 ≈0.003 死区（伸手起步信号缺失）。新核剖面：表面 5cm→0.34、贴面→0.71（渐进不饱和）。
    dists = _fingertip_cube_center_dists(env)                     # (N,5) 指尖到质心距离（不扣半径）
    # [2026-09-16] S2-α4 精核（近端加陡）已回退——"改现有项值分布"引发 value 失配崩；
    #   保留纯宽核（原版行为）。
    per = 1.0 - torch.tanh(dists / touch_std)                     # (N,5) 每指接近度
    # [2026-09-14 四指共享适配] 5 指等权 → 拇指组/四指组 0.5/0.5：
    #   旧式四指占 4/5 权重（同通道动作影响 4 根指尖 → 梯度 4:1 偏四指），与"两侧对置"不对等。
    thumb, four = _thumb_four_split(per)
    touch = 0.5 * thumb + 0.5 * four                              # (N,) 两组平均接近度 0~1
    d = _tcp_cube_dist(env)
    gate = torch.clamp((dist_threshold + d_std - d) / d_std, min=0.0, max=1.0)
    return gate * touch


def finger_contact_reward(
    env: ManagerBasedRLEnv,
    force_std: float = 0.06,
    dist_threshold: float = 0.05,
    d_std: float = 0.02,
) -> torch.Tensor:
    """(2026-09-07 连续化实验) 指尖接触——连续力（tanh）+ EMA 平滑。

    从 0/1 滞回锁存改为连续力，消除阈值边缘 flip-flop 的离散跳变：
      - tanh(f/σ)：σ=0.06，0.05N→0.69、0.10N→0.93、0.15N→0.99
      - EMA α=0.15：滤接触力高频抖动（历史 force_cont 失败教训之一 = 无平滑）
      聚合 0.8·mean + 0.2·min：整体接触力为主、最弱手指为辅助约束，减少单指接触波动影响。
      [2026-09-14 四指共享适配] 聚合改为"拇指组+四指组"：均值为组间 0.5/0.5、min 为组间 min。

    [2026-09-07 失稳回退] σ 曾 0.10（用户建议）：但当前接触力工作点在 0.1N 附近，
      恰好是 tanh(f/0.10) 的梯度最大区（grad=4.2），接触力抖动被放大 3 倍 → value loss 0.004→0.72
      （10665~10667 日志实证）。回退 σ=0.06：f=0.1N 处 grad=1.28，工作点落入饱和区抖动被压死。
      EMA α 0.3→0.15 同步加强平滑。weight 4.0→3.5 补偿 tanh 值回升（3.5×0.93≈3.26 分数不降）。
    ⚠️ 若 value loss 仍 >0.1 或尖峰再现，立即回退二值版（git 历史）。
    """
    from .observations import fingertip_contact_force
    forces = fingertip_contact_force(env)                         # (N,5) N

    # EMA 平滑（滤高频抖动）
    if not hasattr(env, "_contact_force_ema") or env._contact_force_ema.shape[0] != env.num_envs:
        env._contact_force_ema = forces.clone()
    fema = env._contact_force_ema
    fema = 0.15 * forces + 0.85 * fema
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf == 0
        if torch.any(reset_mask):
            fema[reset_mask] = 0.0
    env._contact_force_ema = fema

    per = torch.tanh(fema / force_std)                            # (N,5) 0~1 连续
    # [2026-09-14 四指共享适配] 聚合从 5 指改为"拇指组 + 四指组"：
    #   mean 项 → 组间 0.5/0.5（消除四指 4/5 梯度偏置）；
    #   min 项 → 组间 min（拇指侧/四指侧都要达标，与 grip 两侧 min 同构；旧 min(5) 被几何残差钉死）。
    thumb, four = _thumb_four_split(per)
    contact_mean = 0.5 * thumb + 0.5 * four                       # (N,) 两组平均接触度
    min_contact = torch.minimum(thumb, four)                      # (N,) 两侧短板
    score = 0.8 * contact_mean + 0.2 * min_contact                # (N,) 整体为主 + 短板辅助

    d = _tcp_cube_dist(env)
    gate = torch.clamp((dist_threshold + d_std - d) / d_std, min=0.0, max=1.0)
    return gate * score


def contact_persistence_reward(
    env: ManagerBasedRLEnv,
    force_thresh: float = 0.03,
    ema_alpha: float = 0.35,
    dist_threshold: float = 0.05,
    d_std: float = 0.02,
) -> torch.Tensor:
    """(2026-09-13) 接触持续性（占空比）奖励——专治"指尖贴面颤动"。

    问题（用户 play 慢放观察）：手指接触 cube 时"有时接触有时不接触、变化很快、肉眼可见"，
    cube 并未被推飞——是指面高频断触（contact chatter）。
    现有奖励对颤动是"盲"的：finger_contact EMA α=0.15（≈7 步）、grip EMA α=0.3（≈3 步）
    会把秒级以内的断续抹平；且 TensorBoard 读数为时间平均（1.2N×50% + 0×50% ≈ 0.6N×100%）
    → 策略没有梯度消除"飞快断续"。

    本项设计（对"事件"敏感、对"噪声"鲁棒）：
      ① 瞬时二值化：指尖力 > force_thresh 视为"接触中"——0.03N 远高于传感器噪声，
         不会把接触力高频噪声引入奖励（这是区别于 finger_contact 的关键）；
      ② 短窗 EMA（α=0.35，时间常数 ≈2.9 步 ≈0.1s）→ "近期接触占空比" ∈ [0,1]，
         能抓住 10Hz 级断续；持续贴合 → →1；颤动 → 在 0~1 间波动（均值被砍）；
      ③ 聚合 0.8·mean + 0.2·min（与 finger_contact 一致：拇指组/四指组两组，整体为主、短板辅助）；
      ④ 软门控 d<5cm（姿势就位才计分——"学好姿势准备给力"的阶段之前恒 0 不添乱）。
    与 finger_contact 分工：它管"力多大"（贴住+给力），本项管"别断"（持续贴合）。
    [防 hack] "轻贴不发力"可拿本项分，但拿不到 grip/finger_contact 的力度分——分工明确。
    [可调] 若观察到"贴住但松力"，可乘 tanh(f_ema/0.05~0.1) 力因子；force_thresh 可 ±0.01 微调。
    """
    from .observations import fingertip_contact_force
    forces = fingertip_contact_force(env)                        # (N,5) N
    touching = (forces > force_thresh).float()                   # (N,5) 瞬时"接触中"（事件级）

    # 短窗占空比 EMA（独立状态，与 finger_contact 的 _contact_force_ema 不互扰）
    if not hasattr(env, "_contact_duty_ema") or env._contact_duty_ema.shape != touching.shape:
        env._contact_duty_ema = touching.clone()
    duty = env._contact_duty_ema
    duty = ema_alpha * touching + (1.0 - ema_alpha) * duty
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf == 0
        if torch.any(reset_mask):
            duty[reset_mask] = 0.0
    env._contact_duty_ema = duty

    # [2026-09-14 四指共享适配] 同 finger_contact：拇指组/四指组 0.5/0.5 + 组间 min
    thumb, four = _thumb_four_split(duty)
    per = 0.8 * (0.5 * thumb + 0.5 * four) + 0.2 * torch.minimum(thumb, four)   # (N,)
    d = _tcp_cube_dist(env)
    gate = torch.clamp((dist_threshold + d_std - d) / d_std, min=0.0, max=1.0)
    return gate * per


def contact_alignment_reward(
    env: ManagerBasedRLEnv,
    force_thresh: float = 0.03,
    d_threshold: float = 0.05,
    d_std: float = 0.02,
) -> torch.Tensor:
    """(2026-09-19 力封闭配方) 接触力方向沿夹持轴（cube 局部 y）的程度 → (N,)。

    背景（硬编程实测）：拇指/中指常以"斜擦棱角"方式接触（v0.12：拇指角接触带斜下压分量；
    v0.21："轻触稳、深压沿棱滑脱"）——这种接触能骗到 finger_contact/grip 的分数，
    但夹持轴方向的分量小、竖直分量大，提起时摩擦力的可用方向不对 → "抓得住、提不起"。
    本项给"正向接触"（力矢量指向/背离夹持轴）直接付款，把接触几何往"可提起"引导。

    设计：
      - align = Σ|f_y,i| / Σ|f_i|，逐指在 cube 局部系分解，力度加权（只统计 |f|>force_thresh 的指）：
        0 = 纯斜向/竖直（拿不到分）、1 = 纯对向。无接触指不参与（不稀释/不虚高）。
      - 拇指侧与四指侧力方向相反，但取绝对值 → 两侧同口径（与 grip 的 min 互补：
        grip 管"两侧都够力"，本项管"力方向正确"）。
      - TCP 软门控（与 finger_contact 同款）：悬停区才计分，接近阶段恒 0 不添乱。
      - 加法零项：接触前/纯斜擦 ≈ 0，无负梯度、不与既有项打架。回退：weight 置 0。
    """
    from isaaclab.utils.math import quat_apply_inverse
    forces = _fingertip_force_vectors(env)                        # (N,5,3) 世界系
    cube_quat = env.scene["cube_obj"].data.root_quat_w            # (N,4)
    quat_rep = cube_quat.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
    forces_local = quat_apply_inverse(quat_rep, forces.reshape(-1, 3)).reshape(forces.shape)
    mag = torch.norm(forces, dim=-1)                              # (N,5) 力大小
    mask = (mag > force_thresh).float()                           # 有接触的指
    f_y = forces_local[..., 1].abs() * mask                       # (N,5) 夹持轴分量
    den = (mag * mask).sum(dim=-1) + 1e-6
    align = f_y.sum(dim=-1) / den                                 # (N,) 无接触 → 0
    d = _tcp_cube_dist(env)
    gate = torch.clamp((d_threshold + d_std - d) / d_std, min=0.0, max=1.0)
    return gate * align


def clamp_cube_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """[2026-09-20] 每步写回 cube 的 xy/朝向、清零 xy 与角速度——"xy 锁定 + 不可翻转，只留 z"。

    - PhysX lock 属性编码不可用、replicate 下不传播（实测）→ 状态写回是唯一可控路径；
    - 软锁定：接触力/摩擦/对向力全部真实，只是位置每步被拉回（等效固定销）；
    - 作为极小非零权重 RewTerm 注册（绕开 RewardManager 对 weight==0 的跳过）；
    - 删除该项即解锁 xy（第二阶段课程）。
    """
    obj: RigidObject = env.scene["cube_obj"]
    state = obj.data.root_state_w
    pose = state[:, :7].clone()                          # pos3 + quat4
    vel = state[:, 7:].clone()                           # lin3 + ang3
    if not hasattr(env, "_cube_xy_ref") or env._cube_xy_ref.shape[0] != env.num_envs:
        env._cube_xy_ref = pose[:, 0:2].clone()
        env._cube_quat_ref = pose[:, 3:7].clone()
        env._cube_z0 = pose[:, 2].clone()
    # reset 首步刷新参考（为后续 cube 位置随机化课程做准备）
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        rm = env.episode_length_buf <= 1
        if torch.any(rm):
            m = rm.unsqueeze(-1)
            env._cube_xy_ref = torch.where(m, pose[:, 0:2], env._cube_xy_ref)
            env._cube_quat_ref = torch.where(m, pose[:, 3:7], env._cube_quat_ref)
            env._cube_z0 = torch.where(rm, pose[:, 2], env._cube_z0)
    pose[:, 0:2] = env._cube_xy_ref
    pose[:, 2] = torch.clamp(pose[:, 2], min=env._cube_z0 - 0.001)   # 不下沉（等效桌面支撑）
    pose[:, 3:7] = env._cube_quat_ref
    vel[:, 0:2] = 0.0
    vel[:, 3:6] = 0.0
    obj.write_root_pose_to_sim(pose)
    obj.write_root_velocity_to_sim(vel)
    return torch.zeros(env.num_envs, device=env.device)


# ══════════════════════════════════════════════════════════════
# [2026-09-20 v3 简洁抓取链] 接近 → 抓握 → 提起（参考 DexSuite / IsaacLab 官方 Lift）
#
# 设计依据（调研结论）：
#   - DexSuite 的奖励只有 5~6 项：整手 max 距离接近 + "接触门控"的位置/朝向跟踪 + 成功 +
#     动作惩罚；官方 Lift 亦为 5 项。历史 20+ 项精细奖励导致调试困难与局部最优
#     （悬停骗分、四指不弯）——本版按任务链重设为 7 项，职责单一。
#   - 关键机制"接触门控"（DexSuite contacts()）：没有形成对向抓握时，提起/成功奖励 = 0
#     （"避免悬停在目标附近拿到过高奖励"）。本机适配：拇指侧 vs 工作对（中指+无名指）
#     对向力的 min 软门控。食指/小指碰撞已关闭、力恒 0，不参与。
#   - 任务约定：cube 的 xy/旋转已由 clamp_cube_xy 锁定（只留 z），提起链路更干净；
#     后续课程解锁。指腹朝向暂不做（优先能提起）。
# ══════════════════════════════════════════════════════════════

# 工作对（抓握主力）：中指 + 无名指
_WORKING_PAIR = ("middle4", "ring4")


def _working_hand_dists(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N,3) 拇指 + 工作对（中/无名）指尖 → cube 质心距离。"""
    cube = _get_cube_pos(env)
    names = [_FINGERTIP_NAMES[0], _FINGERTIP_NAMES[2], _FINGERTIP_NAMES[3]]
    return torch.stack([torch.norm(_get_fingertip_pos(env, n) - cube, dim=-1) for n in names], dim=-1)


def whole_hand_reach_reward(env: ManagerBasedRLEnv, std: float = 0.3) -> torch.Tensor:
    """(N,) 整手接近奖励 1−tanh(max_i d_i / std)——取"最大"距离逼整手包覆（DexSuite）。

    与旧 finger_reaching 的"平均距离"不同：平均允许某一两根指头凑近骗分；
    max 要求三根参与指尖都靠近物体才得高分。
    """
    d = _working_hand_dists(env).max(dim=-1).values
    return 1.0 - torch.tanh(d / std)


def _contact_gate(env: ManagerBasedRLEnv, scale: float = 0.1) -> torch.Tensor:
    """(N,) 接触门控 0~1：对向力 min(拇指侧, 工作对侧) 的 tanh 软化。

    [DexSuite contacts() 同构] 没形成对向抓握 → 后续提起/成功奖励为 0。
    scale=0.1N：0.05N→0.46、0.1N→0.76、0.26N（提起所需）→0.93、0.5N→0.98。
    """
    F_py, F_ny = _opposition_components(env)
    opp = torch.minimum(F_py, F_ny)
    return torch.tanh(opp / scale)


def contact_gate_reward(env: ManagerBasedRLEnv, scale: float = 0.1) -> torch.Tensor:
    """(N,) "抓住"的即时奖励 = 门控值本身（0~1）。给策略一个先学会对向接触的密集信号。"""
    return _contact_gate(env, scale)


def _lift_gain(env: ManagerBasedRLEnv, max_height: float) -> torch.Tensor:
    """(N,) 净提升比例 0~1：clamp(z − z_start, 0, max_height) / max_height。

    z_start 为该 env 当前 episode 的 cube 初始高度（episode_length_buf<=1 时刷新）。
    """
    obj: RigidObject = env.scene["cube_obj"]
    z = obj.data.root_pos_w[:, 2]
    if not hasattr(env, "_cube_z_start") or env._cube_z_start.shape[0] != env.num_envs:
        env._cube_z_start = z.clone()
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        reset_mask = env.episode_length_buf <= 1
        if torch.any(reset_mask):
            env._cube_z_start = torch.where(reset_mask, z, env._cube_z_start)
    return torch.clamp(z - env._cube_z_start, min=0.0, max=max_height) / max_height


def _contact_alignment(env: ManagerBasedRLEnv, force_thresh: float = 0.03) -> torch.Tensor:
    """(N,) 接触力沿夹持轴（cube 局部 y）的占比（力度加权，0~1）。纯托举(竖直力)→≈0。"""
    from isaaclab.utils.math import quat_apply_inverse
    forces = _fingertip_force_vectors(env)                        # (N,5,3) 世界系
    cube_quat = env.scene["cube_obj"].data.root_quat_w            # (N,4)
    quat_rep = cube_quat.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
    forces_local = quat_apply_inverse(quat_rep, forces.reshape(-1, 3)).reshape(forces.shape)
    mag = torch.norm(forces, dim=-1)                              # (N,5)
    mask = (mag > force_thresh).float()
    f_y = forces_local[..., 1].abs() * mask
    den = (mag * mask).sum(dim=-1) + 1e-6
    return f_y.sum(dim=-1) / den


def _lift_gate(
    env: ManagerBasedRLEnv,
    deadzone: float = 0.06,
    scale: float = 0.2,
    ema_alpha: float = 0.1,
) -> torch.Tensor:
    """(N,) 提起专用门控：**持续**对向力死区 × 力方向对齐 —— 短暂接触/掌心托举拿不到 lift 分。

    [2026-09-20 v3.6] 三层过滤：
      ① 对向力 opp < 0.06N → 0（防"顶到刚过门槛再托起"；0.06 与 0.04kg 提起需求
         0.13N/侧 匹配：0.06→0、0.1→0.20、0.13→0.34、0.2→0.62、0.35→0.90）；
      ② × align（接触力沿夹持轴 y 的占比）：纯竖直托举 ≈0 → 断粮，只有水平对夹才给分；
      ③ 两者相乘：真夹紧（opp≥0.13 且水平）收益完整，其余取巧收益趋零。
    [2026-09-21 v7.5 监控自主·反作弊] 新增 **对向力 EMA（α=0.1，时间常数 ~0.33s）**：
      逐轮/确定性评测实证两类取巧——(a) "抛起"（瞬时接触后 cube 飞高，z 最高 14cm 而
      指尖力≈0）；(b) "掌心托举"（base_link 无传感器力，cube 被抬到 10cm 而 F_thumb=F_mid=0）。
      瞬时门控对 1~2 步的接触给满额 → 抛/托可骗 lift 分。EMA 后：
        持续夹持（≥0.3s）→ ema≈opp，收益不变；瞬时碰/抛/掌心托 → ema 远低于死区 → 0 分。
      与用户要求一致："力足够且稳定，再去把 cube 提起"。回退：ema_alpha=1.0（纯瞬时）。
    """
    F_py, F_ny = _opposition_components(env)
    opp = torch.minimum(F_py, F_ny)
    if not hasattr(env, "_opp_ema") or env._opp_ema.shape[0] != opp.shape[0]:
        env._opp_ema = torch.zeros_like(opp)
    env._opp_ema = ema_alpha * opp + (1.0 - ema_alpha) * env._opp_ema
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        rm = env.episode_length_buf <= 1
        if torch.any(rm):
            env._opp_ema[rm] = 0.0
    g_force = torch.tanh(torch.clamp(env._opp_ema - deadzone, min=0.0) / scale)
    return g_force * _contact_alignment(env)


def palm_cube_rel_pose_penalty(
    env: ManagerBasedRLEnv,
    pos_deadzone: float = 0.005,
    pos_std: float = 0.02,
    ang_deadzone: float = 0.035,
    ang_std: float = 0.175,
    arrive_dist: float = 0.05,
) -> torch.Tensor:
    """[2026-09-22 v7.11 用户要求] 手掌-cube 相对位姿漂移惩罚（软约束，0~1）：
      每个 episode 开始缓存"手掌(base_link_1)在 cube 坐标系下的位置+姿态"，之后每步算漂移：
        位置项 = 1-exp(-(max(Δp-5mm,0)/2cm)²)   ；姿态项 = 1-exp(-(max(Δθ-2°,0)/10°)²)
      返回 0.5*位置项 + 0.5*姿态项（配合 cfg 的 weight=-1.0 → 位置/姿态各 -0.5 上限）。
    目的（用户）: 抓取/提起过程中手掌与 cube 的相对关系保持不变——手指出力、手掌不乱挪；
      提起时只要手与 cube 同步上升则 Δ≈0 不罚（与反托举方向一致）；顺带治"手往 cube 里
      挪导致的中指中节穿模"。带死区，允许毫米级/几度内的自然微调。
    回退: weight=0。
    """
    from isaaclab.utils.math import quat_inv, quat_mul, quat_apply, quat_error_magnitude
    robot: Articulation = env.scene["robot"]
    cube: RigidObject = env.scene["cube_obj"]
    palm_id = _get_body_id(env, "base_link_1")
    p_palm = robot.data.body_link_pos_w[:, palm_id]
    q_palm = robot.data.body_link_quat_w[:, palm_id]
    q_cube_inv = quat_inv(cube.data.root_quat_w)
    p_rel = quat_apply(q_cube_inv, p_palm - cube.data.root_pos_w)
    q_rel = quat_mul(q_cube_inv, q_palm)
    # [v26 接近阶段] 参考改成"到位才缓存"：TCP 距 cube < arrive_dist 的首次才把当前
    #   相对位姿记为参考；接近途中（未到位）不罚（否则会惩罚接近本身）。
    #   近距离起始（现役）首步即到位 → 行为与旧版一致。
    if not hasattr(env, "_palm_rel_p0") or env._palm_rel_p0.shape[0] != p_rel.shape[0]:
        env._palm_rel_p0 = p_rel.clone()
        env._palm_rel_q0 = q_rel.clone()
        env._palm_ref_set = torch.zeros(p_rel.shape[0], dtype=torch.bool, device=p_rel.device)
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        rm = env.episode_length_buf <= 1
        if torch.any(rm):
            env._palm_ref_set[rm] = False
            env._palm_rel_p0[rm] = p_rel[rm]
            env._palm_rel_q0[rm] = q_rel[rm]
    near = _tcp_cube_dist(env) < arrive_dist
    upd = near & (~env._palm_ref_set)
    if torch.any(upd):
        env._palm_rel_p0[upd] = p_rel[upd]
        env._palm_rel_q0[upd] = q_rel[upd]
        env._palm_ref_set[upd] = True
    dp = torch.norm(p_rel - env._palm_rel_p0, dim=-1)
    dth = quat_error_magnitude(q_rel, env._palm_rel_q0)
    pos_term = 1.0 - torch.exp(-((torch.clamp(dp - pos_deadzone, min=0.0) / pos_std) ** 2))
    ang_term = 1.0 - torch.exp(-((torch.clamp(dth - ang_deadzone, min=0.0) / ang_std) ** 2))
    out = 0.5 * pos_term + 0.5 * ang_term
    return out * env._palm_ref_set.float()   # 未到位=0；到位后参考已缓存并生效


def tcp_reach_reward(
    env: ManagerBasedRLEnv,
    sigma: float = 0.10,
) -> torch.Tensor:
    """[v26 接近阶段] TCP→cube 质心的长尾接近奖励 1−tanh(d/σ)（σ=0.10）：
      远场仍有梯度（20cm→0.02、10cm→0.24、2cm→0.80），配合 cube 出生随机化
      学"从远处把 TCP 移向 cube"。到捏取位 d≈1.5cm→0.85，不干扰贴面/捏持。
      weight≈2.0。回退：weight=0。
    """
    d = _tcp_cube_dist(env)
    return 1.0 - torch.tanh(d / sigma)


def cube_motion_penalty(
    env: ManagerBasedRLEnv,
    v_std: float = 0.05,
    w_std: float = 1.0,
) -> torch.Tensor:
    """[v30 xy解锁] cube 稳持惩罚（0~2）：tanh(|v_xy|/0.05) + tanh(|ω|/1.0)。

    背景：解锁 xy/朝向（删除 clamp_cube_xy 的每步写回）后，策略必须学会"别把
      cube 推跑/转开"。本项对 cube 的横向速度与角速度（世界系）做软饱和惩罚：
      静止≈0；被推滑(5cm/s)→0.76；被转(1rad/s)→0.76。配合捏持链给出"稳"的梯度。
      只罚 xy（z 提起不受影响）。weight≈-0.4。回退：weight=0。
    """
    obj: RigidObject = env.scene["cube_obj"]
    v = obj.data.root_lin_vel_w
    w = obj.data.root_ang_vel_w
    vxy = torch.norm(v[:, :2], dim=-1)
    return torch.tanh(vxy / v_std) + torch.tanh(torch.norm(w, dim=-1) / w_std)


def cube_drift_penalty(
    env: ManagerBasedRLEnv,
    d_std: float = 0.02,
) -> torch.Tensor:
    """[v30d xy解锁] cube 横向漂移惩罚：tanh(|xy − xy_起点|/0.02)。

    诊断（v30c 确定性）：xy 自由后能捏能提(+12.7cm)，但 cube 被拖 5.2cm。
    velocity 惩罚管不住"慢速稳态拖动"（v≈2-3cm/s 罚得轻但位移累积），本项直接罚
    相对每集起点的横向位移：1cm→0.46、2cm→0.76、5cm→0.99。weight≈-0.8。
    回退：weight=0。
    """
    obj: RigidObject = env.scene["cube_obj"]
    p = obj.data.root_pos_w
    if not hasattr(env, "_cube_xy_start2") or env._cube_xy_start2.shape[0] != p.shape[0]:
        env._cube_xy_start2 = p[:, :2].clone()
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        rm = env.episode_length_buf <= 1
        if torch.any(rm):
            env._cube_xy_start2[rm] = p[rm, :2]
    dxy = torch.norm(p[:, :2] - env._cube_xy_start2, dim=-1)
    return torch.tanh(dxy / d_std)


def lift_without_grip_penalty(
    env: ManagerBasedRLEnv,
    deadzone: float = 0.02,
    max_height: float = 0.12,
    force_ref: float = 0.2,
) -> torch.Tensor:
    """[2026-09-21 v7.7 监控自主·反作弊] "无夹持却抬升"惩罚：
      cube 离桌(>deadzone) 且 两侧对向力弱(opp→0) → 惩罚越抬越罚。
    背景: EMA 门控只让"掌心托举/抛起"拿不到 lift 分，但它们是零收益的局部最优，
      策略漂过去后不再回来（v7.5/v7.6 评测: model_1050+ 指尖力=0 却抬 10.6cm）。
      本项给负梯度，把"抬升必须靠持续对向力"变成硬约束（与用户要求一致）。
    强度: 满抬(max_height)+零力 → 1.0（×weight）；真夹持(opp≥0.3N) → weak≈0.13 影响可忽略。
    """
    obj: RigidObject = env.scene["cube_obj"]
    z = obj.data.root_pos_w[:, 2]
    if not hasattr(env, "_cube_z_start") or env._cube_z_start.shape[0] != z.shape[0]:
        env._cube_z_start = z.clone()
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        rm = env.episode_length_buf <= 1
        if torch.any(rm):
            env._cube_z_start = torch.where(rm, z, env._cube_z_start)
    gain = torch.clamp(z - env._cube_z_start - deadzone, min=0.0, max=max_height) / max_height
    F_py, F_ny = _opposition_components(env)
    opp = torch.minimum(F_py, F_ny)
    weak = 1.0 - torch.tanh(opp / force_ref)
    return gain * weak


def finger_pose_reference_reward(
    env: ManagerBasedRLEnv,
    finger_idx: tuple | None = None,
    ref_joint_pos: tuple = (0.0, 1.2217, 0.4363, 0.0),
    sigma: float = 0.5,
) -> torch.Tensor:
    """[v10 C·演示式引导] 指尖闭拢的关节参考吸引：1 − tanh(‖q − q_ref‖ / σ)。

    背景：A（长尾核）+B（进度）实测仍推不动"均值闭中指"（prog≈0、gate=0）——从伸展
      起点学 10cm 闭拢是"从零发现长动作段"，开源灵巧手普遍用演示/参考轨迹解决。
    参考 q_ref = 旧捏取预设（m2=1.2217/m3=0.4363，实测指尖距面 0.8cm、可夹可提，
      对应用户目标"贴面"姿态）。性质：
      - 全程有梯度（远离 ~0、到位 ~1），到参考点饱和、不会过卷；
      - 用 L2 联合角度，不锁单关节，允许策略自选分配；
      - 学成后可把 weight 置 0（课程淡出），交棒 surface/contact/lift。
    回退：weight=0。
    """
    robot: Articulation = env.scene["robot"]
    if finger_idx is None:
        finger_idx = ACTIVE_FOUR_INDICES
    all_ids = _get_finger_joint_ids(robot)
    ref = torch.tensor(ref_joint_pos, device=robot.data.joint_pos.device)
    per = []
    for i in list(finger_idx):
        q = robot.data.joint_pos[:, all_ids[i]]
        n = min(q.shape[-1], ref.shape[-1])
        dq = torch.norm(q[:, :n] - ref[:n], dim=-1)
        per.append(1.0 - torch.tanh(dq / sigma))
    return torch.stack(per, dim=0).mean(dim=0)


def tip_progress_reward(
    env: ManagerBasedRLEnv,
    tip_names: tuple | None = None,
    metric: str = "sphere",
) -> torch.Tensor:
    """[v10 B] 指尖闭拢进度（potential-based shaping）：r = Σ_tips (d_{t−1} − d_t)。

    d = 指尖到 cube 表面的径向距离（不截断）。性质：
      - 每步给"正在靠近"付钱（处处有梯度），远离则负；**停着不动恒 0**，无法薅分；
      - 配合 weight=25（1/m）→ 全程闭拢 10cm ≈ 2.5 分，量级与 contact/lift 匹配；
      - episode 首步/重置行 dx=0（避免虚假进度）。
    回退：weight=0。
    """
    cube = _get_cube_pos(env)
    if tip_names is None:
        tip_names = ACTIVE_TIP_NAMES
    d = torch.zeros(env.num_envs, device=cube.device)
    for n in tip_names:
        if metric == "box":
            d = d + _tip_cube_box_dist(env, n)
        else:
            d = d + (torch.norm(_get_fingertip_pos(env, n) - cube, dim=-1) - _CUBE_HALF_SIZE)
    if not hasattr(env, "_tip_d_prev") or env._tip_d_prev.shape[0] != d.shape[0]:
        env._tip_d_prev = d.clone()
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        rm = env.episode_length_buf <= 1
        if torch.any(rm):
            env._tip_d_prev[rm] = d[rm]
    prog = env._tip_d_prev - d
    env._tip_d_prev = d.clone()
    return prog


def lift_height_reward(env: ManagerBasedRLEnv, max_height: float = 0.08, scale: float = 0.2) -> torch.Tensor:
    """(N,) 提起链尾：净提升（8cm 饱和）× 带死区提起门控——没真夹住一分不给。"""
    return _lift_gain(env, max_height) * _lift_gate(env, scale=scale)


def lift_success_reward(env: ManagerBasedRLEnv, max_height: float = 0.08, scale: float = 0.2) -> torch.Tensor:
    """(N,) 提起达标：净提升 ≥ max_height 且真夹住（死区门控）→ 1（持续给，不终止）。"""
    return (_lift_gain(env, max_height) >= 1.0).float() * _lift_gate(env, scale=scale)


def surface_proximity_reward(
    env: ManagerBasedRLEnv,
    margin: float = 0.03,
    tip_names: tuple | None = None,
    kernel: str = "linear",
    sigma: float = 0.10,
    metric: str = "sphere",
) -> torch.Tensor:
    """[2026-09-20 v4.4] 指尖（拇指+中指）→ cube 表面的线性贴近付款。

    诊断：6550 ckpt 时中指尖距表面仅 2cm、关节已弯到 1.22/0.79（限位 1.57，还有余量），
    但 face_reach 的 tanh 核在 2.7cm 处梯度仅 ~0.05/cm，推不动最后 0.3 rad。
    本项在 margin(3cm) 内**线性**付款：d=2cm→0.33、1cm→0.67、贴面→1.0（×weight 2.0）；
    近场梯度为 1/margin=33/m，是 tanh 核的 ~6 倍——专治"最后一厘米"。
    聚合：拇指 0.5 + 中指 0.5（两侧都要贴近）。
    """
    cube = _get_cube_pos(env)
    if tip_names is None:
        tip_names = ACTIVE_TIP_NAMES          # 单一开关：当前拇指+中指；四指实验可传 5 指尖
    per = []
    for n in tip_names:
        if metric == "box":
            d = _tip_cube_box_dist(env, n)
        else:
            d = torch.norm(_get_fingertip_pos(env, n) - cube, dim=-1) - _CUBE_HALF_SIZE
        if kernel == "tanh":
            # [v10 A] 长尾核（IsaacLab Lift 同款）：1−tanh(d/σ)，任意距离都有梯度，无截断死区
            per.append(1.0 - torch.tanh(d / sigma))
        else:
            per.append(torch.clamp((margin - d) / margin, min=0.0, max=1.0))
    return torch.stack(per, dim=0).mean(dim=0)


def pinch_axis_alignment_reward(
    env: ManagerBasedRLEnv,
    gate_std: float = 0.06,
    center_sigma: float = 0.05,
    tip_names: tuple = ("thumb4", "middle4"),   # (拇指侧指尖, 对侧指尖)——四指实验可改
) -> torch.Tensor:
    """[2026-09-21 v4.7 诊断修复] 对置轴对齐：指尖连线（cube 系 xy 投影）与 cube ±y 轴的对齐度，
    近场门控（远场权重趋零）。

    诊断依据（model_10800 几何实测 + 96k 组全关节可行性搜索）：
      - 复位姿态与策略收敛姿态的"拇指-中指连线"都偏离 cube ±y 轴 ~30°（align_y≈0.85/0.88，dx≈3~8cm）；
      - 固定臂姿下随机手部搜索 1.28 万组：拇指距 ±y 面最近仅 4.3cm（够不到）→ 死锁非学习问题；
      - 大角度臂姿 + TCP 重定心搜索（9.6 万组）：最优可达 max(中,拇)=1.27cm → 可行，但需要腕部
        绕竖轴偏航 ~30°（现有奖励无此项梯度；tcp_orientation 惩罚还在反向压制）。
    本项给出"偏航对齐"的直接梯度：align=1-|dx|/|dxy|（30° 偏 → 0.5，对齐 → 1.0），
    × center（连线中点贴 cube 轴线）× gate（两指尖平均质心距近场门控 1-tanh(d/gate_std)）。
    与 face_reach/surface_proximity 不冲突：前者管"两侧各自贴面"，本项管"连线轴对齐+居中"。
    """
    from isaaclab.utils.math import quat_apply_inverse
    cube: RigidObject = env.scene["cube_obj"]
    cq = cube.data.root_quat_w
    cp = cube.data.root_pos_w
    vt = quat_apply_inverse(cq, _get_fingertip_pos(env, tip_names[0]) - cp)
    vm = quat_apply_inverse(cq, _get_fingertip_pos(env, tip_names[1]) - cp)
    d = vt - vm
    n_xy = torch.norm(d[:, :2], dim=-1) + 1e-6
    align = 1.0 - d[:, 0].abs() / n_xy
    mid = 0.5 * (vt + vm)
    lat = torch.sqrt(mid[:, 0] ** 2 + mid[:, 2] ** 2)
    center = torch.exp(-((lat / center_sigma) ** 2))
    dist = 0.5 * (torch.norm(vt, dim=-1) + torch.norm(vm, dim=-1))
    gate = 1.0 - torch.tanh(dist / gate_std)
    return align * center * gate


def middle_bend_reward(
    env: ManagerBasedRLEnv,
    finger_idx: tuple | None = None,
) -> torch.Tensor:
    """[2026-09-20 v4.2] 中指弯曲的直接奖励（tanh 饱和，无门控）。

    诊断：策略把中指掌根节伸直（default 0.6 → 运行 0），中指尖离 cube 8.9cm；
    finger_close 的层级门控又压制部分弯曲的收益 → "伸直不动"死锁。
    本项直接奖励 j2/j3 弯曲：弯 0.3→0.31、0.6→0.54、1.0→0.76（各 0.5 权重 tanh(x/0.6)）。
    弯曲是接触的前提，空中弯曲无风险、允许（时长不限）。
    """
    robot: Articulation = env.scene["robot"]
    if finger_idx is None:
        finger_idx = ACTIVE_FOUR_INDICES       # 单一开关：当前仅中指；四指实验=(1,2,3,4)
    all_ids = _get_finger_joint_ids(robot)
    vals = []
    for i in list(finger_idx):
        q = robot.data.joint_pos[:, all_ids[i]]
        vals.append(0.5 * torch.tanh(q[:, 1] / 0.6) + 0.5 * torch.tanh(q[:, 2] / 0.6))
    return torch.stack(vals, dim=0).mean(dim=0)


def finger_close_reward(
    env: ManagerBasedRLEnv,
    gate_std: float = 0.08,
    max_flex: float = 0.6,
    prox_min: float = 1.0,
    prox_std: float = 0.10,
) -> torch.Tensor:
    """软门控弯曲奖励：TCP 越近，弯曲奖励越强（1-tanh(d/gate_std)），连续无跳变。
    v19: flex 上限 1.0→0.5——防策略无限加压把 cube 挤飞
    （v18@3053 崩溃：峰值握力→cube 被挤出→终止→value loss 爆 57→策略乱抖螺旋）。
    弯到 0.5 即饱和，不再奖励更大力（实测抓握 flex≈0.2，0.5 足够）。
    [2026-09-02] 硬门控→软门控（借鉴 SoftHand）：旧硬门控 (dist<0.08) 在 TCP 边界波动时
      奖励 0↔1 突变 → 抓取时手指"弯一下直一下"高频颤动；1-tanh 连续过渡消除跳变。
      同时门控 8cm→3cm（gate_std=0.03）：真悬停（TCP 距质心 ~1.5cm）才奖励弯曲，
      防接近阶段提前弯曲碰 cube（play 观察：手指还没悬停就开始弯）。
    [2026-09-17 梯次修正] 默认值同步现役参数（gate_std=0.08 / max_flex=0.6）——σ=0.08 使"塑形"
      在接触带（contact 门 7→5cm）外侧先起效（8cm 0.24 / 5cm 0.45 / 2cm 0.76），恢复"先弯后触"
      的抓握时序；"空中空弯"顾虑由 10cm 外的自然衰减（仅 0.15 且继续降）承担，不再需要 0.03 的极窄门。
    [2026-09-17 贴近因子] prox_min<1.0 时启用"卷且贴"耦合：factor = prox_min + (1-prox_min)*prox，
      prox = 指尖贴近度（与 finger_reaching 同口径：质心距离 + 宽核，拇指组/四指组 0.5/0.5）。
      动机：close（卷）与 reaching（近）可分离 → "浅卷+不接近"死锁；耦合后"卷着靠近"吃双收入、
      造出通往接触的桥。数值：浅卷+远(prox≈0.2)→×0.40；凑近(0.5)→×0.63；贴面(≈0.7)→×0.78。
      prox_min=1.0（默认）= 旧行为，一键回退。
    """
    dist = _tcp_cube_dist(env)                                       # (N,)
    gate = 1.0 - torch.tanh(dist / gate_std)                          # 软门：连续 0~1
    flex = _finger_flex(env.scene["robot"])
    close = gate * torch.min(flex, torch.tensor(max_flex, device=flex.device))
    if prox_min < 1.0:
        # [2026-09-17 贴近因子] "卷且贴"耦合：factor = prox_min + (1-prox_min)*prox
        #   prox = 指尖贴近度（与 finger_reaching 同口径：质心距离 + 宽核，拇指组/四指组 0.5/0.5）
        dists = _fingertip_cube_center_dists(env)                     # (N,5)
        per = 1.0 - torch.tanh(dists / prox_std)                      # (N,5)
        thumb, four = _thumb_four_split(per)
        prox = 0.5 * thumb + 0.5 * four                               # (N,) 0~1
        close = close * (prox_min + (1.0 - prox_min) * prox)
    return close

def thumb_opposition_reward(
    env: ManagerBasedRLEnv,
    gate_std: float = 0.05,
) -> torch.Tensor:
    """(2026-09-03) 拇指对侧奖励——引导拇指绕到四指对侧（力封闭的几何前提）。

    play(1800/2000) 诊断：拇指初始就在侧面，策略学会"拇指+四指邻侧"贴面，
    但 grip（基于接触力的对向 min）在拇指绕到对侧之前恒 0，无梯度引导 →
    策略永远卡在邻侧，力封闭（grip）恒 0。

    本项提供密集几何引导（接触前就有梯度）：
      opp = -dot(normalize(thumb-cube)_xy, normalize(fingers_mean-cube)_xy)
      对侧→1，邻侧→0，同侧→-1（clamp 到 0，只奖不罚，避免与 finger_contact 打架）。

    设计要点（引导效果）：
    - 只看水平面 x-y：对侧夹持是水平对向，含 z 会误奖"拇指压顶"作弊；
    - clamp(min=0)：同侧/邻侧给 0 不罚，策略不会因"暂时没到位"被额外惩罚而不敢探索；
    - TCP 软门控 1-tanh(d/gate_std)：接近阶段不激活，悬停后（d~1.5cm）才引导拇指绕位；
    - [2026-09-04] 参照改回"四指 mean"（_FINGERTIP_NAMES[1:]）：此前食指中指 mean 偏，
      拇指对侧方向被拉偏；绕位已稳定，改回四指中心作为对向面基准。
    - [2026-09-09] thumb4→thumb2→thumb4：thumb2（侧摆段）只反映 thumb1+2 绕位，拇指指尖仍在侧面
      （play 观察）→ 改回 thumb4 让指尖/指腹真正绕到对侧。历史 thumb4 的"弯曲掉分"问题在 50g 轻弯
      （max_flex=0.2）下已大幅缓解（拇指弯曲小，thumb4 偏离对侧幅度小）。
    """
    cube = _get_cube_pos(env)                                   # (N,3)
    thumb = _get_fingertip_pos(env, "thumb4")                   # (N,3) 拇指指尖/指腹（[2026-09-09] thumb2→thumb4：thumb2 对侧但指尖仍侧面，改回指尖真正绕到对侧）
    fingers = torch.stack(
        [_get_fingertip_pos(env, name) for name in _FINGERTIP_NAMES[1:]], dim=0
    ).mean(dim=0)                                               # (N,3) 四指平均（对向面中心）

    v_thumb = thumb[:, :2] - cube[:, :2]                        # 水平分量 (N,2)
    v_fingers = fingers[:, :2] - cube[:, :2]
    v_thumb = v_thumb / (torch.norm(v_thumb, dim=-1, keepdim=True) + 1e-6)
    v_fingers = v_fingers / (torch.norm(v_fingers, dim=-1, keepdim=True) + 1e-6)
    opp = -torch.sum(v_thumb * v_fingers, dim=-1)               # (N,) 对侧 1 / 邻侧 0 / 同侧 -1

    d = _tcp_cube_dist(env)
    gate = 1.0 - torch.tanh(d / gate_std)                       # TCP 近才激活
    return gate * torch.clamp(opp, min=0.0)

def _fingers_side_hysteresis(
    env: ManagerBasedRLEnv,
    v_f_y: torch.Tensor,
    margin: float = 0.005,
) -> torch.Tensor:
    """工作对所在侧符号（±1），带滞回 → (N,)。

    [2026-09-19 目标稳定] 原实现 s = sign(v_f.y)：工作对悬在 cube 顶面/中轴附近时
    v_f.y≈0，符号随抖动翻转 → four_face_reach 与 thumb_face_reach 的目标点在 ±y 两面
    之间跳变，两侧引导自相矛盾。滞回规则：|v_f.y| > margin 才更新符号，滞回带内保持
    历史值。状态存 env._s_four_side（两个函数共享，保证"同侧/对面"口径一致）。
    """
    s_now = 2.0 * (v_f_y >= 0.0).float() - 1.0
    if not hasattr(env, "_s_four_side") or env._s_four_side.shape[0] != v_f_y.shape[0]:
        env._s_four_side = s_now.clone()
    else:
        confident = v_f_y.abs() > margin
        env._s_four_side = torch.where(confident, s_now, env._s_four_side)
    return env._s_four_side


def thumb_face_reach_reward(
    env: ManagerBasedRLEnv,
    sigma: float = 0.025,
    gate_std: float = 0.08,
    finger_name: str | None = None,            # "工作对"指尖（默认单一开关的第 2 个）
) -> torch.Tensor:
    """(2026-09-16 S2-α7) 拇指专项"最后 1cm"——拇指尖→对侧面中心 的紧核接近奖励。

    背景（实证链）：opposition 交棒后（α6），play 仍见"拇指 3/4 不卷、不贴面"；
      GUI 手拖四个拇指关节证实指尖**可达**对侧面（非结构死区）→ 缺的是"末段密集
      付款 + 协同动作引导"。close 的 min 只管"别落后于四指"（还有 gate/cap 截断），
      本项把"贴到那一厘米"变成单项付款：离目标点越近分越高。

    设计（门把手式目标 + 防作弊）：
      - 目标点 = 对侧面中心：先取"四指均值相对质心"在 cube 局部 y（grasp 轴，与
        grip/_opposition_components 同轴）上的符号 s_four，目标局部坐标 (0, −s_four·half, 0)。
        到"四指对面"只有一个门把手——摸顶/邻侧/悬空都拿不到分。
      - 距离核 1−tanh(d/σ)，σ=2.5cm：只覆盖最后一小段（2cm→0.34、1cm→0.62、贴面→≈0.8+）。
      - TCP 软门控（与 close 同款）：悬停区才计分，接近阶段≈0 不干扰。
      - 单指、无 min、无跨指聚合：不会像 close 的 min 那样先补四指；也不会像
        opposition 那样对卷曲反付钱（本项对卷曲只会加钱）。
    """
    from isaaclab.utils.math import quat_apply_inverse
    cube = _get_cube_pos(env)                                   # (N,3)
    # [2026-09-16 修复] 本机副本此前漏掉 cube_quat 定义（v_f/v_t 直接引用它 → 一启用就 NameError）；
    #   与 _opposition_components 同款取法补回。
    cube_quat = env.scene["cube_obj"].data.root_quat_w          # (N,4)
    # [2026-09-20 v4] 只用中指：无名指碰撞已关（用户资产调整，仅拇指+中指参与），
    #   其穿模位置会带偏均值。
    fingers = _get_fingertip_pos(env, finger_name or ACTIVE_TIP_NAMES[1])  # (N,3) 工作对指尖
    thumb = _get_fingertip_pos(env, _FINGERTIP_NAMES[0])        # (N,3) 拇指尖
    v_f = quat_apply_inverse(cube_quat, fingers - cube)         # (N,3) 中指（cube 局部系）
    v_t = quat_apply_inverse(cube_quat, thumb - cube)           # (N,3) 拇指（cube 局部系）
    s_four = _fingers_side_hysteresis(env, v_f[:, 1])           # (N,) 四指侧（±1，带滞回防跳变）
    target = torch.zeros_like(v_t)
    target[:, 1] = -s_four * _CUBE_HALF_SIZE                    # 对侧面中心（局部坐标）
    d = torch.norm(v_t - target, dim=-1)                        # (N,) 到目标点距离
    reach = 1.0 - torch.tanh(d / sigma)                         # (N,) 紧核 0~1
    d_tcp = _tcp_cube_dist(env)
    gate = 1.0 - torch.tanh(d_tcp / gate_std)                   # TCP 近才激活
    return gate * reach

def four_face_reach_reward(
    env: ManagerBasedRLEnv,
    sigma: float = 0.05,
    gate_std: float = 0.08,
    finger_name: str | None = None,            # "工作对"指尖（默认单一开关的第 2 个）
) -> torch.Tensor:
    """(2026-09-17) 四指专项"最后 1cm"——工作对（中指+无名指）指尖 → 四指同侧面中心 的紧核接近奖励。

    背景（用户 play 实证）：地板 0.12 + contact 2.5 后，四指已恢复"钩爪状"卷曲，
      但停在面外不贴合（"钩爪悬空"）——最后 1~2cm 只有 reaching 宽核远尾 + close 的
      prox 微坡（合计 ~0.1~0.2/步/cm），而"贴上面"的大额付款（contact，四指加入后
      ≈+1.2/步）藏在"必须先真接触"的冷启动门槛（及 cube 被扰动的稳定税）后面。
      → "卷满+不碰"是当前理性局部最优。本项把"走到那一厘米"变成单独付款
      （镜像 thumb_face_reach——它在拇指上已把"悬空对置"治到半贴面）。

    设计（门把手式目标 + 防作弊，与拇指版同构）：
      - 目标点 = 四指同侧面中心：工作对（[2:4]，与 contact/_opposition 同口径）在
        cube 局部 y（grasp 轴）上的符号 s_four → 局部坐标 (0, +s_four·half, 0)。
        摸顶/邻侧/悬空都拿不到靶心分。
      - 距离核 1−tanh(d/σ)，σ=5cm：5cm→0.24、3cm→0.46、2cm→0.62、1cm→0.80；
        与 reaching 宽核在末段叠加成"最后一厘米"的连续坡度。
      - TCP 软门控（与 close 同款）：悬停区才计分，接近阶段≈0 不干扰。
      - 位置项、无 min、不管卷曲：不与 close（关节角）/contact（力）争口径；
        对"卷着贴"只会加钱，不对任何形态反向惩罚。
      - 权重 0 一键回退；接触前/远处≈0（"加零项"）。
    """
    from isaaclab.utils.math import quat_apply_inverse
    cube = _get_cube_pos(env)                                   # (N,3)
    cube_quat = env.scene["cube_obj"].data.root_quat_w          # (N,4)
    # [2026-09-20 v4] 只用中指：无名指碰撞已关，穿模位置会带偏均值。
    fingers = _get_fingertip_pos(env, finger_name or ACTIVE_TIP_NAMES[1])  # (N,3) 工作对指尖
    v_f = quat_apply_inverse(cube_quat, fingers - cube)         # (N,3) 中指（cube 局部系）
    s_four = _fingers_side_hysteresis(env, v_f[:, 1])           # (N,) 四指侧（±1，带滞回防跳变）
    target = torch.zeros_like(v_f)
    target[:, 1] = s_four * _CUBE_HALF_SIZE                     # 同侧面中心（局部坐标）
    d = torch.norm(v_f - target, dim=-1)                        # (N,) 到目标点距离
    reach = 1.0 - torch.tanh(d / sigma)                         # (N,) 紧核 0~1
    d_tcp = _tcp_cube_dist(env)
    gate = 1.0 - torch.tanh(d_tcp / gate_std)                   # TCP 近才激活
    return gate * reach



# ══════════════════════════════════════════════════════════════
# Sparse Reward: 成功
# ══════════════════════════════════════════════════════════════

def success_stage1_reward(
    env: ManagerBasedRLEnv,
    palm_dist_threshold: float = 0.06,
    lin_vel_threshold: float = 0.05,
    ang_vel_threshold: float = 0.10,
    lin_vel_std: float = 0.05,
    ang_vel_std: float = 0.05,
    d_std: float = 0.02,
    orient_threshold: float | None = None,
    z_std: float | None = None,
    contact_fade_floor: float | None = None,
    contact_fade_scale: float = 0.1,
) -> torch.Tensor:
    """Stage 1 成功: TCP-Cube 距离 < palm_dist_threshold 且物体稳定（不要求手指）。
    逐步发放，episode 不终止。

    [2026-09-19 交棒门控 grip_fade_floor] 非 None 时启用"接触后 success 淡出"：
      对向夹持力 opp 越大，success 因子越向 grip_fade_floor 收敛
      （fade = 1-(1-floor)·tanh(opp/scale)，opp=0→1，opp→∞→floor）。
      动机（历史实证）：贴面/压紧必然推动 cube → success 的"稳定"从满值直落 0，
      "碰 cube"变成亏本买卖 → 策略的最优解是"悬停但永不贴面"（09-18/09-19 两次 run
      停在 four_face 0.15~0.44 后崩）。淡出后：贴住过程的扰动惩罚大幅降低，
      力封闭的收益（grip/contact/align）可以覆盖；同时保留 floor 底线防"完全放弃悬停"。
      建议 floor 0.3、scale 0.2（opp=0.2N→fade≈0.47，opp=0.5N→fade≈0.31）。
      回退：grip_fade_floor=None（行为与旧版完全一致）。

    ═══ 当前版本（激活）：饱和型（d_std 过渡带）+ z 方向约束 + 软 stable ═══
    v87: 二值→饱和型——消除 0/1 硬跳变对价值函数的冲击。
    [2026-09-12 平稳化] stable 同步软化：v_lin/v_ang 越过阈值后在 std 过渡带内线性衰减
      （旧版 0/1 硬切在阈值边缘 flip → success 跳 4.0 → value 冲击，当前 value loss 0.13 超警戒线）。
    v93: orient_threshold 平放姿态门控（env_cfg 不传则不启用）。
    [用户] z_std: TCP 只允许在 cube 质心及以上——质心以上满分，质心下方 z_std 内线性衰减到 0
      （禁从下方接近/越过质心；贴面 TCP 在质心下方 3cm → 0）。线性衰减避免硬跳变。
    切换回最初版本（二值 0/1）时：启用下方注释段，并注释掉上面的 return。

    ═══ 最初版本（v25 时代，注释保留）：二值（0/1）═══
    """
    obj: RigidObject = env.scene["cube_obj"]
    # [2026-09-11 解冲突] v_lin 只看 xy 速度（z 上升=目标运动，不破坏 stable）。
    #   旧版 3D 范数含 v_z：策略一抬升 cube → v_lin 超阈值 → stable=0 → success 掉 3 分，
    #   反向激励"不动"，与 lift（奖励抬升）目标冲突 → lift 恒 0 的结构性原因之一。
    v_lin = torch.norm(obj.data.root_lin_vel_w[:, :2], dim=-1)
    v_ang = torch.norm(obj.data.root_ang_vel_w, dim=-1)
    d = _tcp_cube_dist(env)
    close = torch.clamp((palm_dist_threshold + d_std - d) / d_std, min=0.0, max=1.0)
    close_raw = close  # [2026-09-17 诊断] 纯距离因子（不乘 above），仅供日志
    if z_std is not None:
        # [用户] 方向约束：TCP z >= cube 质心 z → 满分；质心下方 z_std 内线性衰减 → 0
        tcp_z = _get_tcp_pos(env)[:, 2]
        cube_z = _get_cube_pos(env)[:, 2]
        above = torch.clamp(1.0 + (tcp_z - cube_z) / z_std, min=0.0, max=1.0)
        close = close * above
    # [2026-09-12 平稳化] 硬二值 → 软判定（v87 对 close 做过、stable 漏了）：
    #   旧版 0/1 硬切：cube 被抓取推动时速度在阈值边缘抖动 → stable 0/1 flip → success 跳 4.0
    #   → value loss 冲击（当前 0.13 超警戒线 0.10 的最大来源）。
    #   新版：阈值内满分，越过阈值在 std 过渡带内线性衰减到 0（连续无跳变）。
    v_lin_score = 1.0 - torch.clamp((v_lin - lin_vel_threshold) / lin_vel_std, 0.0, 1.0)
    v_ang_score = 1.0 - torch.clamp((v_ang - ang_vel_threshold) / ang_vel_std, 0.0, 1.0)
    stable = v_lin_score * v_ang_score
    # [2026-09-17 诊断·零行为] success 四因子写 TB（Episode_Reward/succ_*；只记日志、不进奖励、不加权）。
    #   读法：success 读数 = weight × succ_close × succ_above × succ_stable（均为时间平均）。
    #   succ_close 卡 0.6~0.75 平 → "悬停停浅"（d 落在 close 斜坡 1.5~3.5cm 中段，看 succ_d 深度）；
    #   succ_close≈0.95+ 而 success 仍低 → "到达占时"（20cm 行程吃掉太多时间）或 stable/above 掉。
    #   机制复刻 opp_py 的 _episode_sums 日志（RewardManager 泛型遍历，自动进 extras/TB）。
    # if hasattr(env, "reward_manager"):
    #     sums = env.reward_manager._episode_sums
    #     if "succ_close" not in sums:
    #         sums["succ_close"] = torch.zeros_like(close_raw)
    #         sums["succ_d"] = torch.zeros_like(d)
    #         sums["succ_stable"] = torch.zeros_like(stable)
    #     sums["succ_close"] += close_raw * env.step_dt
    #     sums["succ_d"] += d * env.step_dt
    #     sums["succ_stable"] += stable * env.step_dt
    #     if z_std is not None:
    #         if "succ_above" not in sums:
    #             sums["succ_above"] = torch.zeros_like(above)
    #         sums["succ_above"] += above * env.step_dt
    result = close * stable
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
        result = result * oriented
    # [2026-09-19 干预5 交棒修正] 触发从"对向力 opp"改为"任一指尖接触力"：
    #   旧实现用 opp=min(F+y,F-y)，单侧轻碰（学接触的必经阶段）时 opp 不涨 → success 不淡出
    #   → "轻碰 = 扰动 cube + success 掉分"仍全额成立 → 接触恐惧（手指长期悬停在 cube 顶上方
    #   5.6cm 不下探的直接原因之一）。改用 max 指尖接触力：任何一侧碰到就开始淡出。
    if contact_fade_floor is not None:
        from .observations import fingertip_contact_force
        fmax = fingertip_contact_force(env).max(dim=-1).values          # (N,) 任一指尖接触力
        fade = 1.0 - (1.0 - contact_fade_floor) * torch.tanh(fmax / contact_fade_scale)
        result = result * fade
    return result

    # ---- 最初版本（切换时启用本段，并注释掉上面的 return）----
    # obj: RigidObject = env.scene["cube_obj"]
    # v_lin = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    # v_ang = torch.norm(obj.data.root_ang_vel_w, dim=-1)
    # d = _tcp_cube_dist(env)
    # close = (d < palm_dist_threshold).float()
    # stable = ((v_lin < lin_vel_threshold) & (v_ang < ang_vel_threshold)).float()
    # return close * stable


def success_stage2_reward(
    env: ManagerBasedRLEnv,
    touch_std: float = 0.008,
    force_std: float = 0.02,
    grip_scale: float = 0.1,
) -> torch.Tensor:
    """Stage 2 成功（抓取成型）：五指贴面 × 五指有力 × 对向夹持，饱和型 0~1。

    与 Stage 1（TCP 接近）区分：本项确认"真抓取"——
      1) touch：五指指尖到 cube 面的真实距离达标（与 finger_contact 的 touch 同口径）；
      2) force：五指接触力达标（sensor 有数据 → 证明指尖真正压上 cube 面，而非悬空）；
      3) grip：cube 局部 y 方向对向力达标（与 opposition_reward 同算法 → 证明夹住而非单侧推）。
    三者乘积：任一不满足即 0，逼策略同时满足"贴面、有力、对向"。
    饱和型（tanh/exp 软过渡）而非二值：避免 0/1 硬跳变冲击 value loss。
    [2026-09-04] 新增，供 Stage 2 抓取阶段替换 success_stage1（env_cfg 切阶段时换 func）。
    """
    from .observations import fingertip_contact_force

    # 1) 指组贴面（touch 两组均值）→ 0~1　[2026-09-14 四指共享适配：拇指组/四指组 0.5/0.5]
    dists = _fingertip_cube_surface_dists(env)                       # (N,5)
    t_thumb, t_four = _thumb_four_split(1.0 - torch.tanh(dists / touch_std))
    touch = 0.5 * t_thumb + 0.5 * t_four                             # (N,)

    # 2) 指组有力（force 两组均值）→ 0~1　[同上适配]
    forces = fingertip_contact_force(env)                            # (N,5)
    f_thumb, f_four = _thumb_four_split(1.0 - torch.exp(-forces / force_std))
    force = 0.5 * f_thumb + 0.5 * f_four                             # (N,)

    # 3) 对向夹持（grip）→ 0~1：cube 局部 y 方向对向力（与 opposition_reward 同算法，不门控）
    from isaaclab.utils.math import quat_apply_inverse
    fvec = _fingertip_force_vectors(env)                             # (N,5,3) 世界系
    cube_quat = env.scene["cube_obj"].data.root_quat_w               # (N,4)
    quat_rep = cube_quat.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
    fy = quat_apply_inverse(quat_rep, fvec.reshape(-1, 3)).reshape(fvec.shape)[..., 1]
    F_py = torch.relu(fy).sum(dim=-1)
    F_ny = torch.relu(-fy).sum(dim=-1)
    grip = torch.tanh(torch.minimum(F_py, F_ny) / grip_scale)        # (N,) 0~1

    return touch * force * grip


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


def hand_action_magnitude_penalty(
    env: ManagerBasedRLEnv,
    gate_dist: float | None = None,
) -> torch.Tensor:
    """(v96) 惩罚手指动作幅度——逼手指输出 0（静止）。

    hand_action_penalty 只罚"弯曲量"（一阶矩），手指仍可在"不弯"的前提下抖动（高熵）。
    本项直接罚策略输出平方（二阶矩），唯一低惩罚解 = 手指输出 0（确定性静止），
    给无目标的手指通道提供确定性约束，根治 entropy_coef 熵奖励下的手指熵漂移/熵爆。
    动作拼接顺序：arm_action(6D OSC) 在前，hand_action(8D GroupedHandAction) 在后。
    Stage 2 激活 finger_contact 时需调低本项（让手指能动抓取）。

    [2026-09-03] 加 TCP 距离门控 gate_dist：抓取奖励（finger_close gate_std=0.03 等）在
      TCP 距 Cube > gate_dist 时≈0 → 手指通道无梯度 → 熵漂移 → 熵爆（本次 entropy 10.3）。
      门控后：远→1 罚（逼手指 0 防熵漂移），近→0 放开（抓取阶段手指自由弯曲）。
      （与 hand_action_penalty 的 gate_dist 同语义：远罚近放）
    """
    action = env.action_manager.action  # (N, 14)
    hand = action[:, 6:]                # 手指 8 维（拇指4 + 四指共享4）
    val = torch.clamp(torch.sum(torch.square(hand), dim=-1), max=10.0)
    if gate_dist is not None:
        # 距离门控：远→1（罚动作幅度逼 0），近→0（放开给抓取）
        d = _tcp_cube_dist(env)
        val = val * (d > gate_dist).float()
    return val


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
    # v97: clamp max=100——物理瞬态时 speed 爆表（数 m/s）→ pen=(speed/0.05)² 可到数万，
    #   单步 reward -0.05×数万 污染 value（value loss 149036 崩溃的直接元凶）。
    #   正常接近 speed<0.3m/s（pen<36），clamp 100 只截断物理爆炸，不干扰正常训练。
    return torch.clamp(pen, max=100.0)


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

    # v97: clamp max=100——平方无上界（ang=π 时 pen≈438），探索期手腕瞬态大偏离会污染 value。
    #   正常偏离<40°（pen<30）、探索 60°（pen~160）→ clamp 100 保留正常+探索惩罚，只截断极端。
    # [2026-09-14 漂移修复] clamp 100→30：配合 weight -0.05→-0.2（env_cfg），
    #   单步惩罚上限保持 ≤30×0.2=6（与旧 100×0.05=5 相当，不增尖峰），
    #   但 0.2~1rad 区间的防漂移力度 ×4。
    return torch.clamp((ang / ang_std) ** 2, max=30.0)


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


def _opposition_components(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,) 两侧对向力分量 [F_+y, F_-y]（cube 局部 y 两侧力分别求和，未取 min，raw 未平滑）。

    供 _opposition_force（取 min）复用。
    """
    from isaaclab.utils.math import quat_apply_inverse
    forces = _fingertip_force_vectors(env)                    # (N,5,3) 世界系
    cube_quat = env.scene["cube_obj"].data.root_quat_w        # (N,4)
    quat_rep = cube_quat.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)   # (N*5, 4)
    forces_flat = forces.reshape(-1, 3)                       # (N*5, 3)
    forces_local = quat_apply_inverse(quat_rep, forces_flat).reshape(forces.shape)  # (N,5,3) cube 局部系
    fy = forces_local[..., 1]                                 # (N,5) 局部 y 分量
    F_py = torch.relu(fy).sum(dim=-1)                         # (N,) +y 侧对向力
    F_ny = torch.relu(-fy).sum(dim=-1)                        # (N,) -y 侧对向力
    return F_py, F_ny


# ──────────────────────────────────────────────────────────────
# [2026-09-14 S2 激活] 本函数已恢复启用（S1 期间曾整体注释）。
#   调用方：opposition_reward（grip 的 func）、lift_reward —— 两处共用，勿再注释。
# ──────────────────────────────────────────────────────────────
def _opposition_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N,) 对向夹持力 = min(F_+y, F_-y)（取两侧较弱侧，强制两侧都够力才给高分）。

    [2026-09-06] 抓取姿势是「侧面夹持」（拇指/四指沿 cube 局部 y 对顶），不是托举：
      提起 cube 靠夹持力的摩擦力（f=μ·F_y），故直接判据是对向夹持力 opp，不是 z 向托力。
      从 opposition_reward 抽出，供 grip 奖励与 lift 门控复用。
    min 的物理意义：稳态时两侧力必平衡（牛顿第三定律）；min 取弱侧 = 链条最弱环，
      opp≥0.98 等价于"两侧都≥0.98"，逼策略把弱侧也加够，从而两侧平衡且摩擦力 2μF≥mg。
    """
    F_py, F_ny = _opposition_components(env)
    # [2026-09-10] DexManus 版 metric 记录：RewardManager 用 _episode_sums 累积、reset() 时写成
    #   extras["Episode_Reward/<name>"]（扁平键，÷max_episode_length_s）。直接累积 F_py×step_dt，
    #   曲线数值≈力值(N)，且不进入总 reward。此前 env.extras 嵌套键方案在该版本不存在（KeyError 根因）。
    #   ⚠️ 放大倍数 = 本函数每步的调用次数（动态）：当前仅 grip 调用（×1，读数=时间平均实际值）；
    #   开 lift 后 ×2；grip_force 已注释。读 metric 前先数一遍当前启用的调用方。
    sums = env.reward_manager._episode_sums
    if "opp_py" not in sums:
        sums["opp_py"] = torch.zeros_like(F_py)
        sums["opp_ny"] = torch.zeros_like(F_ny)
    sums["opp_py"] += F_py * env.step_dt
    sums["opp_ny"] += F_ny * env.step_dt
    # [2026-09-13 颤动诊断] 单侧断触时间占比：瞬时力 < 0.05N 的帧计 1 × dt 累积（episode 平均 = 占比 0~1）。
    #   用户 play 观察：接触颤动、贴不住，且 opp_py/opp_ny 曲线"连续平滑"——因为它们是时间平均
    #   （1.2N×50% + 0×50% 的均值 = 0.6N×100% 的均值，颤动被平均掩盖）。
    #   本指标专补盲区：读法 = 该侧"无接触"的时间比例。
    #   对照读：avg力高 + 断触率高 = 断续大力（颤动）；avg力高 + 断触率低 = 稳定贴合。
    #   （接近阶段天然为断触 → 前段值天然偏高，学会抓握后应骤降。）
    if "opp_break_py" not in sums:
        sums["opp_break_py"] = torch.zeros_like(F_py)
        sums["opp_break_ny"] = torch.zeros_like(F_ny)
    sums["opp_break_py"] += (F_py < 0.05).float() * env.step_dt
    sums["opp_break_ny"] += (F_ny < 0.05).float() * env.step_dt
    # [2026-09-13 颤动诊断②] 断开次数：从"接触中"(≥0.05N) 跌落到"无接触"的事件计数（每 episode）。
    #   与"占比"的关键区别：接近阶段（从未接触）天然不计数——这是纯"接触后断掉"的颤动频率，
    #   不受接近段时长稀释，无需门控。读法：次数/10s → ÷10 = 次/秒（如 50 → 每秒断 5 次）。
    #   组合判读：占比高 + 次数高 = 高频碎断（策略颤动，治策略）；
    #            占比高 + 次数低 = 单次长断（几何/姿势/力不平衡问题，治结构）。
    contact_py_now = F_py >= 0.05
    contact_ny_now = F_ny >= 0.05
    if not hasattr(env, "_opp_contact_prev") or env._opp_contact_prev[0].shape != contact_py_now.shape:
        env._opp_contact_prev = (contact_py_now.clone(), contact_ny_now.clone())
    prev_py, prev_ny = env._opp_contact_prev
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf == 0
        if torch.any(reset_mask):   # reset 帧把 prev 对齐当前值，不产生虚假断开事件
            prev_py = torch.where(reset_mask, contact_py_now, prev_py)
            prev_ny = torch.where(reset_mask, contact_ny_now, prev_ny)
    if "opp_breaks_py" not in sums:
        sums["opp_breaks_py"] = torch.zeros_like(F_py)
        sums["opp_breaks_ny"] = torch.zeros_like(F_ny)
    sums["opp_breaks_py"] += (prev_py & ~contact_py_now).float()
    sums["opp_breaks_ny"] += (prev_ny & ~contact_ny_now).float()
    env._opp_contact_prev = (contact_py_now, contact_ny_now)
    # [2026-09-10] 两侧施力点 z 高度（诊断倾倒力矩根因：对向力不共线 → τ=F·Δz 翻倒 cube）。
    #   四指指尖 z 均值 vs 拇指指尖 z。Δz = z_fingers − z_thumb（两条曲线之差 ÷3 = 实际米数）。
    #   独立初始化（不依赖 opp_py 分支，防后续增删 key 时 KeyError）。
    z_fingers = torch.stack(
        [_get_fingertip_pos(env, n)[:, 2] for n in _FINGERTIP_NAMES[1:]], dim=0
    ).mean(dim=0)                                                        # (N,) 四指指尖 z 均值
    z_thumb = _get_fingertip_pos(env, _FINGERTIP_NAMES[0])[:, 2]        # (N,) 拇指指尖 z
    if "z_fingers" not in sums:
        sums["z_fingers"] = torch.zeros_like(z_fingers)
        sums["z_thumb"] = torch.zeros_like(z_thumb)
    sums["z_fingers"] += z_fingers * env.step_dt
    sums["z_thumb"] += z_thumb * env.step_dt
    return torch.minimum(F_py, F_ny)                          # (N,) 对向夹持力


def opposition_reward(
    env: ManagerBasedRLEnv,
    gate_dist: float = 0.015,
    scale: float = 0.21,
    d_std: float = 0.01,
) -> torch.Tensor:
    """(v52) 对向夹持奖励（力封闭的平滑形式）——专治"四指挤同侧、无法夹"。

    背景（v51 诊断）：四指全压 +x 面 → 所有力同向 → 无对向 → cube 侧滑。
    grip_force（总力）会被单侧猛压骗到高分，无法区分"抓得稳"与"单侧推"。

    [2026-09-04] 世界系 → cube 局部系，只取局部 y 轴对向：
      抓握方向 = "手轴→四指方向" = cube 局部 y，拇指/四指沿 y 对向夹持；
      x 方向无手指抓握（侧摆方向），剔除 x 力——避免侧摆挤压的 x 向力被误计入 grip。
      对向度量：R = min(F_+y, F_-y)，再 tanh 饱和到 0~1。

    [2026-09-12 回退] 曾加 balance_gate×align_gate 做成"完整力封闭"，但最大单项 weight 3~4 的结构
      大改 → value 失配 → 续训一开始 reward 全崩（奖励项归零只剩惩罚）。回退纯 tanh(min)。
      "力相等/等高"改由独立项 opposition_balance / opposition_height_align 承担（权重小、可单独调）。
    """
    # ⚠️ [2026-09-14] _opposition_force 当前整段被注释（S1 停用）——启用本项(grip)前先恢复它！
    opp = _opposition_force(env)                              # (N,) 对向夹持力（cube 局部 y）
    # [2026-09-05] EMA 平滑：接触力每物理步抖动，scale=0.05 会放大噪声 → 对 raw 对向力做指数移动平均，
    #   滤掉高频抖动，grip 曲线更稳。alpha=0.3（约 3~5 帧平滑，滞后可忽略）；reset 时用当前值初始化。
    if not hasattr(env, "_grip_opp_ema") or env._grip_opp_ema.shape[0] != env.num_envs:
        env._grip_opp_ema = opp.clone()
    ema = env._grip_opp_ema
    ema = 0.3 * opp + 0.7 * ema
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf == 0
        if torch.any(reset_mask):
            ema[reset_mask] = 0.0
    env._grip_opp_ema = ema
    opp = torch.tanh(ema / scale)                              # 0→1 平滑饱和
    dist = _tcp_cube_dist(env)
    # v71: 硬门控 → 软门控（d_std 过渡带）；grip 项复用本函数做"正确抓取=力封闭"确认
    gate = torch.clamp((gate_dist + d_std - dist) / d_std, min=0.0, max=1.0)
    return gate * opp


def opposition_balance_reward(
    env: ManagerBasedRLEnv,
    opp_thresh: float = 0.40,
    engage_steep: float = 10.0,
) -> torch.Tensor:
    """(2026-09-10) 两侧对向力平衡奖励——逼弱侧（拇指侧）追上强侧（四指侧）。

    opp = min(F_py, F_ny) 的结构性盲区：min 只由弱侧决定，减强侧不改变 opp，
      grip/grip_force 完全无感 → 策略可"四指单侧猛推 + 拇指弱接触"拿满 grip 分，
      但两侧不平衡 → 四指持续推 cube → 接触滑动 + 提不起。
    本项显式奖励"两侧平衡"，给"拇指侧加力/增接触"提供直接梯度。

    两点设计（[2026-09-10] 用户建议：奖励得分应纯由平衡决定，不被其他因素污染）：
      - balance = 1 − |F_py−F_ny|/(F_py+F_ny)：相对不平衡度，0=单侧 1=平衡。唯一主因子。
      - engaged = sigmoid(engage_steep×(F_weak−opp_thresh))：夹持建立（弱侧>0.4N）才激活。
        接触前两侧力都 0（balance=1 假平衡）→ engaged≈0 压死，避免接近阶段白拿满分。

    [移除 level 因子] 旧版 level=tanh(F_weak/force_std) 原意是堵"松四指侧"作弊（松强侧→
      balance↑但 level 不变），但它让得分随"力大小"漂移、读数不是纯平衡度、梯度不纯。
      "逼弱侧涨"的职责已由 grip（逼 min 涨）+ grip_force（平衡后逼两侧一起涨）接管，
      level 是重复劳动。移除后本项读数直接 = 平衡度（夹持建立后），可当诊断指标用。
    """
    F_py, F_ny = _opposition_components(env)                 # (N,) 两侧对向力（raw）
    F_weak = torch.minimum(F_py, F_ny)                       # (N,) 弱侧 = opp
    total = F_py + F_ny
    imbalance = torch.abs(F_py - F_ny) / (total + 1e-6)      # (N,) 0~1
    balance = 1.0 - imbalance                                # (N,) 1=平衡
    engaged = torch.sigmoid(engage_steep * (F_weak - opp_thresh))   # (N,) 夹持建立软门
    return engaged * balance


def opposition_height_align_reward(
    env: ManagerBasedRLEnv,
    opp_thresh: float = 0.40,
    z_std: float = 0.02,
    engage_steep: float = 10.0,
) -> torch.Tensor:
    """(2026-09-10→11) 施力点高度对齐到 cube 质心——治"蜷缩勾棱边" + 消除倾倒力矩。

    [2026-09-11 升级] 从"两侧互相对齐"（Δz=z_fingers−z_thumb→0）升级为"两侧都锚定到质心高度"：
      蜷缩的空间特征 = 指尖在 cube 顶面以上勾棱边（实测 z_fingers 0.819 > 质心 0.78 + 半高 0.03）。
      旧版只逼两侧相等，两侧一起升高（都在上方勾棱边）也能拿满分，治不了蜷缩。
      新版 dz_f=z_fingers−cube_z、dz_t=z_thumb−cube_z 分别锚定质心：两侧都在质心高度
      = 共线（零力矩）+ 过质心（指腹贴侧面，不勾棱边）——一个约束同时治蜷缩和倾倒。

    设计要点：
      - align = exp(−(dz_f/σ)²)·exp(−(dz_t/σ)²)：两侧都贴质心才满分。正 shaping 无负尖峰。
      - engaged = sigmoid(engage_steep×(F_weak−opp_thresh))：夹持建立才激活，接近阶段不约束。
      - 空间位置约束（指尖 z 坐标），非关节约束 → 避开手指通道锁死雷区。
      - cube 被提起后 cube_z 上升，指尖应跟着走（相对质心不变），动态锚定天然成立。
    """
    F_py, F_ny = _opposition_components(env)                 # (N,) 两侧对向力（raw）
    F_weak = torch.minimum(F_py, F_ny)                       # (N,) 弱侧 = opp
    cube_z = _get_cube_pos(env)[:, 2]                        # (N,) cube 质心 z
    z_fingers = torch.stack(
        [_get_fingertip_pos(env, n)[:, 2] for n in _FINGERTIP_NAMES[1:]], dim=0
    ).mean(dim=0)                                            # (N,) 四指指尖 z 均值
    z_thumb = _get_fingertip_pos(env, _FINGERTIP_NAMES[0])[:, 2]   # (N,) 拇指指尖 z
    dz_f = z_fingers - cube_z                                # (N,) 四指相对质心高度
    dz_t = z_thumb - cube_z                                  # (N,) 拇指相对质心高度
    align = torch.exp(-(dz_f / z_std) ** 2) * torch.exp(-(dz_t / z_std) ** 2)   # (N,) 都贴质心=1
    engaged = torch.sigmoid(engage_steep * (F_weak - opp_thresh))   # (N,) 夹持建立软门
    return engaged * align


# ══════════════════════════════════════════════════════════════
# 举升（Stage 3）
# ══════════════════════════════════════════════════════════════

def lift_reward(
    env: ManagerBasedRLEnv,
    gate_dist: float = 0.05,
    gate_steep: float = 10.0,
    flex_thresh: float = 0.2,
    flex_steep: float = 5.0,
    z_thresh: float = 0.78,
    lift_std: float = 0.01,
    lift_force_thresh: float = 0.03,
    force_steep: float = 100.0,
) -> torch.Tensor:
    """(Stage 3) Cube 离桌 1~2cm 的举升奖励。门控: TCP 贴 cube + 手指弯曲 + 对向夹持力足够。

    [2026-09-06 重设计] 原版 z_thresh=1.25（举到半米高）对“提 1~2cm 离桌”目标太稀疏，
    策略永远拿不到分学不会。几何：桌面顶 0.75、cube 半高 0.03、初始中心 0.78。
      - [2026-09-11] z_thresh=0.78（cube 贴桌质心）：一离桌就给分，消除 0.78~0.79 死区。
        tanh(lifted/0.01) 让离桌 1cm≈0.76、2cm≈0.96（软饱和无尖峰）。
      - gate_dist 0.3→0.05：TCP 必须贴着 cube 才给分，防“没抓住就抬手”作弊。
      - [2026-09-06] 加夹持力门控：gripped = sigmoid(100×(opp−0.03))。姿势是侧面夹持（不是托举），
        提起靠夹持摩擦（f=μ·F_y），故门控用对向夹持力 opp（y 向两侧取 min），不用 z 向托力。
        没夹住（opp→0）→ gripped≈0 → lift 压死，杜绝“贴面未抓稳就空抬”破坏抓取（play 8500/8600 诊断）。
        注：门控只防空抬；真正夹起还需 opp > mg/(2μ)=0.164N（mass=0.05kg、μ=1.5），
        这由 cube_z 实际升高隐含要求。（旧注释 0.59N 是 mass=0.18kg 时代残留，已修正）
      - 原 sqrt(lift) 在 0 附近导数→∞ 会放大噪声，改 tanh 导数有界。
    """
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["cube_obj"]
    close = torch.sigmoid(gate_steep * (gate_dist - _tcp_cube_dist(env)))
    flexed = torch.sigmoid(flex_steep * (_finger_flex(robot) - flex_thresh))
    # ⚠️ [2026-09-14] _opposition_force 当前整段被注释（S1 停用）——启用本项前先恢复它！
    opp = _opposition_force(env)                                 # (N,) 对向夹持力（侧面夹持的直接判据）
    gripped = torch.sigmoid(force_steep * (opp - lift_force_thresh))  # (N,) 夹得够紧才给 lift 分
    # [2026-09-11 消死区] 不 clamp：z_thresh=0.78=cube 贴桌质心，cube 一离桌（z>0.78）就有梯度。
    #   旧版 clamp(z−0.79) 在 z∈[0.78,0.79] 死区无梯度，策略无引导跨不过 1cm → lift 恒 0。
    lifted = obj.data.root_pos_w[:, 2] - z_thresh               # (N,) 离桌高度（贴桌=0，负=被压）
    return close * flexed * gripped * torch.tanh(lifted / lift_std)


def object_goal_tracking_reward(
    env: ManagerBasedRLEnv,
    lift_std: float = 0.02,
    table_height: float = 0.75,
    hand_gate_dist: float = 0.06,
    hand_gate_std: float = 0.03,
) -> torch.Tensor:
    """(2026-09-13 方案一+撞飞修复) 提起奖励 —— 纯"离桌高度"版（与 command 解耦）+ 手-物体门控。

    [撞飞 hack 修复] 崩溃证据：机械臂乱动把 hand 带离 cube、cube 被撞飞 → clearance>0 →
    """
    from isaaclab.utils.math import matrix_from_quat

    obj: RigidObject = env.scene["cube_obj"]

    # 最低顶点离桌间隙（支撑函数）：翻滚不会抬高最低顶点，只有真提起才会
    R = matrix_from_quat(obj.data.root_quat_w)                   # (N, 3, 3)
    z_offset = (R[:, 2, 0].abs() + R[:, 2, 1].abs() + R[:, 2, 2].abs()) * _CUBE_HALF_SIZE
    bottom_z = obj.data.root_pos_w[:, 2] - z_offset              # 立方体最低顶点 z
    clearance = torch.clamp(bottom_z - table_height, min=0.0)    # 离桌间隙（平放=0，压入=0）

    # [撞飞 hack 修复] 手-物体距离门控：手离 cube >6cm 时得分快速衰减到 0
    d = _tcp_cube_dist(env)
    hand_gate = 1.0 - torch.tanh(torch.clamp(d - hand_gate_dist, min=0.0) / hand_gate_std)

    return torch.tanh(clearance / lift_std) * hand_gate           # 离桌高度 × 手-物体门控


def object_goal_bonus_once(
    env: ManagerBasedRLEnv,
    clearance_threshold: float = 0.15,
    bonus: float = 20.0,
    table_height: float = 0.75,
    hand_dist_max: float = 0.08,
) -> torch.Tensor:
    """(2026-09-13 方案一+撞飞修复) 提起达标一次性奖励 —— 与 command 解耦 + 手-物体条件。

    判定：clearance（最低顶点离桌间隙）> 0.15（离桌 15cm）**且手在 cube 8cm 内**。
    [撞飞 hack 修复] 崩溃证据：cube 被机械臂乱动撞飞（腾空>15cm）→ bonus 被触发（10 分）
    → 强化"乱动"。加 hand_dist_max 条件后：撞飞时手被带离（d 大）→ 不触发；
    真提起（手抓着，d 1.5~5cm）→ 正常触发。
    用 clearance 而非质心 z：翻滚不触发（同 tracking 的防 hack 判定）。
    严格一次性（到达后锁定，掉落再到达不重复发），reset 事件清除（reset_goal_bonus_granted）。
    """
    from isaaclab.utils.math import matrix_from_quat

    obj: RigidObject = env.scene["cube_obj"]

    R = matrix_from_quat(obj.data.root_quat_w)                   # (N, 3, 3)
    z_offset = (R[:, 2, 0].abs() + R[:, 2, 1].abs() + R[:, 2, 2].abs()) * _CUBE_HALF_SIZE
    bottom_z = obj.data.root_pos_w[:, 2] - z_offset              # 立方体最低顶点 z
    clearance = torch.clamp(bottom_z - table_height, min=0.0)    # 离桌间隙

    d = _tcp_cube_dist(env)
    reached = (clearance > clearance_threshold) & (d < hand_dist_max)   # 高 + 手在附近

    N = env.num_envs
    if not hasattr(env, "_goal_bonus_granted") or env._goal_bonus_granted.shape[0] != N:
        env._goal_bonus_granted = torch.zeros(N, dtype=torch.bool, device=env.device)
    first_time = reached & ~env._goal_bonus_granted
    reward = first_time.float() * bonus
    env._goal_bonus_granted = env._goal_bonus_granted | reached
    return reward


def reset_goal_bonus_granted(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """reset 回调：清除指定 envs 的 goal 到达标记。"""
    N = env.num_envs
    if not hasattr(env, "_goal_bonus_granted") or env._goal_bonus_granted.shape[0] != N:
        env._goal_bonus_granted = torch.zeros(N, dtype=torch.bool, device=env.device)
    env._goal_bonus_granted[env_ids] = False
