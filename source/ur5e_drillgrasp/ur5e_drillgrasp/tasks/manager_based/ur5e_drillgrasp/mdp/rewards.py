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
from .observations import _BODY_OFFSET, _CUBE_HALF_SIZE, _FINGERTIP_NAMES, _get_body_id


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


def _thumb_four_split(per: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """每指量 (N,5) → (拇指组 (N,), 四指组 (N,))。手指顺序 = _FINGERTIP_NAMES（[0]拇指、[2:4]=工作对）。

    [2026-09-16 捏持化·工作对] 四指组改取"中指+无名指"（[2:4]）：食指/小指碰撞已在资产里关闭
      （不能与 cube 交互、力信号恒零）——若沿用 [1:5] 均值，四指组被两个零稀释一半（天花 1.0→0.5）、
      contact 的"拇指入场"诊断（>1.2）随之失真。改 [2:4] 后天花与诊断口径与旧版一致。
    """
    return per[:, 0], per[:, 2:4].mean(dim=-1)

# 每指 4 关节权重 [侧摆1, 掌根2, 中段3, 指尖4]（跨指聚合 0.7·min+0.3·mean 见 _finger_flex）
_FINGER_JOINT_WEIGHTS = [
    torch.tensor([0.8, 0.5, 1.5, 1.0]),  # 拇指
    torch.tensor([0.0, 1.2, 0.8, 0.0]),  # 食指
    torch.tensor([0.0, 1.2, 0.8, 0.0]),  # 中指
    torch.tensor([0.0, 1.2, 0.5, 0.0]),  # 无名指
    torch.tensor([0.0, 1.2, 0.5, 0.0]),  # 小指
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
            # 层级门控：j3 计分需 j2 弯 >0.15，j4 计分需 j3 弯 >0.15
            g3 = torch.sigmoid(8.0 * (q[:, 1] - 0.60))
            g4 = torch.sigmoid(8.0 * (q[:, 2] - 0.15))
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
    thumb, four = per[0], per[2:4].mean(dim=0)               # 各 (N,)
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

def thumb_face_reach_reward(
    env: ManagerBasedRLEnv,
    sigma: float = 0.025,
    gate_std: float = 0.08,
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
    # [2026-09-16 捏持化·工作对] 侧判定用工作对（中指+无名指）——食指/小指无碰撞，穿模时位置会带偏均值
    fingers = torch.stack(
        [_get_fingertip_pos(env, n) for n in _FINGERTIP_NAMES[2:4]], dim=0
    ).mean(dim=0)                                               # (N,3) 工作对均值（中指+无名指）
    thumb = _get_fingertip_pos(env, _FINGERTIP_NAMES[0])        # (N,3) 拇指尖
    v_f = quat_apply_inverse(cube_quat, fingers - cube)         # (N,3) 四指（cube 局部系）
    v_t = quat_apply_inverse(cube_quat, thumb - cube)           # (N,3) 拇指（cube 局部系）
    s_four = 2.0 * (v_f[:, 1] >= 0.0).float() - 1.0             # (N,) 四指侧（±1）
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
    # [2026-09-16 捏持化·工作对] 食指/小指无碰撞（穿模时位置会带偏均值）→ 用工作对（中指+无名指）
    fingers = torch.stack(
        [_get_fingertip_pos(env, n) for n in _FINGERTIP_NAMES[2:4]], dim=0
    ).mean(dim=0)                                               # (N,3) 工作对均值
    v_f = quat_apply_inverse(cube_quat, fingers - cube)         # (N,3) 工作对（cube 局部系）
    s_four = 2.0 * (v_f[:, 1] >= 0.0).float() - 1.0             # (N,) 四指侧（±1）
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
) -> torch.Tensor:
    """Stage 1 成功: TCP-Cube 距离 < palm_dist_threshold 且物体稳定（不要求手指）。
    逐步发放，episode 不终止。

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
