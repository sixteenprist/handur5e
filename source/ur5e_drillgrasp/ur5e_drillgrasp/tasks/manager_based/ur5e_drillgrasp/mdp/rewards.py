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
    """(W) 五指尖→Cube 表面距离 → (N,5)。扣除了 Cube 半边长。每指独立。

    [2026-09-07] 曾改精确边界框距离修小指/无名指颤动，但精确距离让小指/无名指“贴棱边即满分”→
      finger_reaching 在棱边处梯度消失→手指 12 维对应维度无梯度→熵漂移快速崩。
      暂回退球近似（|指尖-质心|-半边长）：颤动问题后置，先专注逼力主线。
    """
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
# 拇指 [对掌1, 侧摆2, 中段3, 指尖4] = [0.8, 0.2, 1.5, 0.2]
#   [Stage 2 2026-09-02] 对掌1 权重 1.2→0.8：finger_close 奖励对掌弯曲，thumb1 朝下弯会
#   碰 cube 上表面（play 观察物理阻挡）。降权减少"对掌弯曲"的正向驱动，仍保留部分（抓握需要）。
# v18 崩溃根因 = 只有掌根(2)弯、中段(3)不弯 → 指尖够不到、只有压力没接触 → 挤飞
_FINGER_JOINT_WEIGHTS = [
    torch.tensor([0.8, 0.5, 1.5, 0.8]),  # 拇指: 对掌1 / 侧摆2 / 中段3 / 指尖4（[2026-09-04] 指尖4 0.2→0.5：加强拇指指尖弯曲）
    torch.tensor([0.0, 1.2, 0.8, 0.05]),
    # 食指: 侧摆1 0.2→0.0（[2026-09-04] 剔除侧摆）；指尖4 0.2→0.05（[2026-09-07] 治指尖戳 cube，改用指腹贴）
    torch.tensor([0.0, 1.2, 0.8, 0.05]),  # 中指
    torch.tensor([0.0, 1.2, 0.5, 0.05]),  # 无名指
    torch.tensor([0.0, 1.2, 0.5, 0.05]),  # 小指
]


def _finger_flex_per_finger(robot: Articulation) -> torch.Tensor:
    """每根手指的加权弯曲度量 → (5, N)。0=伸直, ~1.57=全弯。
    （从 _finger_flex 抽出，供手指一致性奖励等复用）
    """
    abs_pos = torch.abs(robot.data.joint_pos)
    per_finger = []
    for i, ids in enumerate(_get_finger_joint_ids(robot)):
        w = _FINGER_JOINT_WEIGHTS[i].to(abs_pos.device)  # (4,)
        q = abs_pos[:, ids]  # (N,4) [j1,j2,j3,j4]
        if i > 0 and len(ids) >= 4:
            # 层级门控：j3 计分需 j2 弯 >0.15，j4 计分需 j3 弯 >0.15
            g3 = torch.sigmoid(8.0 * (q[:, 1] - 0.30))
            g4 = torch.sigmoid(8.0 * (q[:, 2] - 0.30))
            contrib = w[0] * q[:, 0] + w[1] * q[:, 1] + w[2] * q[:, 2] * g3 + w[3] * q[:, 3] * g4
            per_finger.append(contrib / w.sum())
        else:
            per_finger.append((q * w).sum(dim=1) / w.sum())
    return torch.stack(per_finger, dim=0)  # (5, N)


def _finger_flex(robot: Articulation) -> torch.Tensor:
    """手指弯曲度量 (rad)。0=伸直, ~1.57=全弯。
    关节加权 + 层级门控（v41）：
    - 四指 掌根2 权重 1.2——根节先弯、弯得多
    - 后续关节在上一关节已弯（>0.15）的基础上才计分：j3 计分需 j2 弯，j4 计分需 j3 弯
      （sigmoid 软门）——自然的手指顺序弯曲，防"指尖独弯的蜷缩爪"
    - 拇指保持原权重，不做层级门控（解剖不同）
    跨指 0.7×最不弯 + 0.3×平均，防拇指独自弯完。
    """
    per = _finger_flex_per_finger(robot)  # (5, N)
    return 0.7 * per.min(dim=0).values + 0.3 * per.mean(dim=0)


def finger_sync_reward(
        env: ManagerBasedRLEnv,
        std: float = 0.2,
        dist_threshold: float = 0.02,
        d_std: float = 0.01,
) -> torch.Tensor:
    """(SoftHand 借鉴) 四指弯曲量一致性奖励——治"小指/无名指独弯、食/中指不弯"。

    play 诊断：策略驱动小指/无名指（共享通道）多弯、食指/中指几乎不弯，
    而 _finger_flex 的 0.7·min+0.3·mean 能被"单侧多弯 + 另一侧弯一点"骗过。
    本项按四指弯曲量的方差给 exp(-var/std)：四指弯曲越同步分越高，
    任何一根明显落后/超前都压分——逼"所有手指一起弯"。
    [用户] 排除拇指：拇指对掌/弯曲姿势与其余四指天然不同，不参与一致性。
    [v70] std 0.03→0.2：0.03 太严——四指差异 0.3rad 时 exp(-0.09/0.03)=0.05 且梯度≈0，
      奖励恒 0 学不动。0.2 下差异 0.3→0.64、0.5→0.29，连续梯度引导逐步同步。
    带 TCP 距离软门控（1cm 内满分，1~2cm 线性过渡，与 success 一致）。
    """
    per = _finger_flex_per_finger(env.scene["robot"])[1:]  # (4, N) 排除拇指(第0行)
    diff = per - per.mean(dim=0, keepdim=True)
    variance = diff.square().mean(dim=0)  # (N,) 各指不一致程度
    sync = torch.exp(-variance / std)  # (N,) 0~1
    d = _tcp_cube_dist(env)
    gate = torch.clamp((dist_threshold + d_std - d) / d_std, min=0.0, max=1.0)
    return gate * sync


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
    forces = _fingertip_force_vectors(env)  # (N,5,3) 世界系
    down = torch.clamp(-forces[..., 2].sum(dim=-1), min=0.0)  # (N,) 向下合力
    if deadzone > 0.0:
        down = torch.clamp(down - deadzone, min=0.0)
    press = torch.tanh(down / force_std)  # (N,) 0~1 软饱和，无尖峰
    d = _tcp_cube_dist(env)
    gate = (d < gate_dist).float()
    return gate * press


def palm_proximity_penalty(
        env: ManagerBasedRLEnv,
        surface_threshold: float = 0.02,
        std: float = 0.01,
) -> torch.Tensor:
    """手掌距 cube 表面太近惩罚——防"压近手掌"hack finger_contact 的 touch。

    Stage 2 观察（2026-09-01）：finger_contact 只按"指尖→表面距离"计分，
    策略学会"把手掌压近让指尖贴面"而不弯曲手指 → 手掌从悬停(2cm)变贴压(<2cm)
    → 连锁导致 thumb1 掌根段侵入 cube 上方空间、碰到 cube 上表面（物理阻挡）。
    本项：手掌表面距离 < surface_threshold 时平方惩罚，逼手掌保持悬停，
    让 finger_contact 的贴面只能靠"手指弯曲"实现。
    """
    palm_dist = _palm_cube_dist(env)  # 手掌→cube 质心欧氏距离
    surface_dist = palm_dist - _CUBE_HALF_SIZE  # 手掌→cube 表面距离
    violation = torch.clamp(surface_threshold - surface_dist, min=0.0)
    return (violation / std) ** 2


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
        touch_std: float = 0.03,
        dist_threshold: float = 0.15,
        d_std: float = 0.05,
) -> torch.Tensor:
    """(2026-09-04) 指尖接近 cube 面奖励——密集梯度（bootstrap 用）。

    与 finger_contact（纯力判据）分离：本项只奖励"指尖靠近 cube 面"，
    宽过渡带（touch_std=3cm）提供从远到近的连续梯度，避免稀疏。
    不奖励"贴面有力"——那由 finger_contact 的纯力 + 滞回判据负责。
    [拆分原因] 原 touch×force 混合结构：touch 几何距离在"接近"即饱和 → 白拿底分，
      而 force 在"接触"才激活 → "接近→接触"梯度断档。拆成"距离引导 + 力判据"两段。
    """
    dists = _fingertip_cube_surface_dists(env)  # (N,5) 指尖到表面距离（已扣半边长）
    touch = (1.0 - torch.tanh(dists / touch_std)).mean(dim=-1)  # (N,) 平均接近度 0~1
    d = _tcp_cube_dist(env)
    gate = torch.clamp((dist_threshold + d_std - d) / d_std, min=0.0, max=1.0)
    return gate * touch


def finger_contact_reward(
        env: ManagerBasedRLEnv,
        force_std: float = 0.06,
        dist_threshold: float = 0.15,
        d_std: float = 0.05,
) -> torch.Tensor:
    """(2026-09-07 连续化实验) 指尖接触——连续力（tanh）+ EMA 平滑。

    从 0/1 滞回锁存改为连续力，消除阈值边缘 flip-flop 的离散跳变：
      - tanh(f/σ)：σ=0.06，0.05N→0.69、0.10N→0.93、0.15N→0.99
      - EMA α=0.15：滤接触力高频抖动（历史 force_cont 失败教训之一 = 无平滑）
      聚合 0.8·mean + 0.2·min：整体接触力为主、最弱手指为辅助约束，减少单指接触波动影响。

    [2026-09-07 失稳回退] σ 曾 0.10（用户建议）：但当前接触力工作点在 0.1N 附近，
      恰好是 tanh(f/0.10) 的梯度最大区（grad=4.2），接触力抖动被放大 3 倍 → value loss 0.004→0.72
      （10665~10667 日志实证）。回退 σ=0.06：f=0.1N 处 grad=1.28，工作点落入饱和区抖动被压死。
      EMA α 0.3→0.15 同步加强平滑。weight 4.0→3.5 补偿 tanh 值回升（3.5×0.93≈3.26 分数不降）。
    ⚠️ 若 value loss 仍 >0.1 或尖峰再现，立即回退二值版（git 历史）。
    """
    from .observations import fingertip_contact_force
    forces = fingertip_contact_force(env)  # (N,5) N

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

    per = torch.tanh(fema / force_std)  # (N,5) 0~1 连续
    contact_mean = per.mean(dim=-1)  # (N,) 平均接触度
    min_contact = per.min(dim=-1).values  # (N,) 短板
    score = 0.8 * contact_mean + 0.2 * min_contact  # (N,) 整体为主 + 短板辅助

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
      ③ 聚合 0.8·mean + 0.2·min（与 finger_contact 一致：整体为主、短板辅助）；
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

    per = 0.8 * duty.mean(dim=-1) + 0.2 * duty.min(dim=-1).values   # (N,)
    d = _tcp_cube_dist(env)
    gate = torch.clamp((dist_threshold + d_std - d) / d_std, min=0.0, max=1.0)
    return gate * per


def finger_close_reward(
        env: ManagerBasedRLEnv,
        gate_std: float = 0.03,
        max_flex: float = 0.5,
) -> torch.Tensor:
    """软门控弯曲奖励：TCP 越近，弯曲奖励越强（1-tanh(d/gate_std)），连续无跳变。
    v19: flex 上限 1.0→0.5——防策略无限加压把 cube 挤飞
    （v18@3053 崩溃：峰值握力→cube 被挤出→终止→value loss 爆 57→策略乱抖螺旋）。
    弯到 0.5 即饱和，不再奖励更大力（实测抓握 flex≈0.2，0.5 足够）。
    [2026-09-02] 硬门控→软门控（借鉴 SoftHand）：旧硬门控 (dist<0.08) 在 TCP 边界波动时
      奖励 0↔1 突变 → 抓取时手指"弯一下直一下"高频颤动；1-tanh 连续过渡消除跳变。
      同时门控 8cm→3cm（gate_std=0.03）：真悬停（TCP 距质心 ~1.5cm）才奖励弯曲，
      防接近阶段提前弯曲碰 cube（play 观察：手指还没悬停就开始弯）。
    """
    dist = _tcp_cube_dist(env)  # (N,)
    gate = 1.0 - torch.tanh(dist / gate_std)  # 软门：连续 0~1
    flex = _finger_flex(env.scene["robot"])
    return gate * torch.min(flex, torch.tensor(max_flex, device=flex.device))


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
    - [2026-09-04] 拇指参考点 thumb4 指尖→thumb2 侧摆段：用指尖度量时 thumb4 伸直即满分、
      thumb3/4 弯曲反而偏离对侧→掉分→策略故意保持拇指指尖伸直。绕到对侧只由 thumb1 对掌
      + thumb2 侧摆决定，改用 thumb2 后弯曲 thumb3/4 不再掉对侧分。
    """
    cube = _get_cube_pos(env)  # (N,3)
    thumb = _get_fingertip_pos(env, "thumb2")  # (N,3) 拇指侧摆段（绕位由 thumb1+2 决定，弯曲 thumb3/4 不影响此点）
    fingers = torch.stack(
        [_get_fingertip_pos(env, name) for name in _FINGERTIP_NAMES[1:]], dim=0
    ).mean(dim=0)  # (N,3) 四指平均（对向面中心）

    v_thumb = thumb[:, :2] - cube[:, :2]  # 水平分量 (N,2)
    v_fingers = fingers[:, :2] - cube[:, :2]
    v_thumb = v_thumb / (torch.norm(v_thumb, dim=-1, keepdim=True) + 1e-6)
    v_fingers = v_fingers / (torch.norm(v_fingers, dim=-1, keepdim=True) + 1e-6)
    opp = -torch.sum(v_thumb * v_fingers, dim=-1)  # (N,) 对侧 1 / 邻侧 0 / 同侧 -1

    d = _tcp_cube_dist(env)
    gate = 1.0 - torch.tanh(d / gate_std)  # TCP 近才激活
    return gate * torch.clamp(opp, min=0.0)


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
            ids.append(j[1])  # 第2个关节 = 掌根
    ids_t = torch.tensor(ids, device=robot.device)
    abs2 = torch.abs(robot.data.joint_pos[:, ids_t])  # (N, len)
    flex2 = 0.5 * abs2.mean(dim=-1) + 0.5 * abs2.min(dim=-1).values  # (N,)
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
    dists = _fingertip_cube_surface_dists(env)  # (N,5) 第0列 = thumb
    per = 1.0 - torch.tanh(dists[:, 0] / std)  # (N,)
    tcp_dist = _tcp_cube_dist(env)
    gate = 1.0 - torch.tanh(tcp_dist / gate_std)  # Soft TCP gate
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
      人手抓取靠中段包络、指尖适度伸直，0.3·min 让它们停在自然    # [2026-09-08 观测] 两侧对向力分量（weight=0，仅 tensorboard 观测，不影响训练）：
    #   opp=min(F_py,F_ny)；两侧稳态应相等，观察两侧差可诊断"单侧推"作弊 / 一侧够一侧虚。
姿态。
    v23: std 0.08→0.04——0.08 时差 1-2cm 就 82%，无"最后 1cm"梯度；
    0.04 只奖励真正贴到，恢复接触梯度（策略已贴近，不再怕尖梯度难学）。
    """
    dists = _fingertip_cube_surface_dists(env)  # (N,5)
    per_finger = 1.0 - torch.tanh(dists / std)  # (N,5)
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
            ids.append(j[3])  # 第4关节 = 指尖
    ids_t = torch.tensor(ids, device=robot.device)
    abs4 = torch.abs(robot.data.joint_pos[:, ids_t])  # (N,5)
    over = torch.clamp(abs4 - thresh, min=0.0).mean(dim=-1)  # (N,)
    dist = _tcp_cube_dist(env)
    gate = (dist < dist_threshold).float()
    return gate * over

def finger_abduction_penalty(
    env: ManagerBasedRLEnv,
    thresh: float = 0.1,
    dist_threshold: float = 0.15,
) -> torch.Tensor:
    """(解耦力与姿势) 手指侧摆（外展/内收）惩罚——治"手指侧向张开推挤邻指/cube"的作弊姿势。

    侧摆关节偏离 0 越远越罚，逼手指在抓取平面内对夹，而非侧向张开。
    GroupedHandAction 关节顺序（play 已证实）：
      四指 [侧摆1, 掌根2, 中段3, 指尖4] → 侧摆 = ids[0]
      拇指 [对掌1, 侧摆2, 中段3, 指尖4] → 侧摆 = ids[1]（thumb2_joint，限位 ±60°）
    仅贴近 cube 时生效（不干扰接近阶段）。
    """
    robot: Articulation = env.scene["robot"]
    ids_list = _get_finger_joint_ids(robot)
    ab_abs = []
    for i, ids in enumerate(ids_list):
        j = ids[1] if i == 0 else ids[0]   # 拇指侧摆=第2关节，四指侧摆=第1关节
        ab_abs.append(torch.abs(robot.data.joint_pos[:, j]))
    abs_ab = torch.stack(ab_abs, dim=-1)                     # (N,5)
    over = torch.clamp(abs_ab - thresh, min=0.0).mean(dim=-1)  # (N,) 超阈值的侧摆量
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
    jp = torch.abs(robot.data.joint_pos)  # (N, num_joints)
    scores = []
    for i, ids in enumerate(_get_finger_joint_ids(robot)):
        if i == 0 or len(ids) <= 3:  # 跳过拇指
            continue
        j3 = jp[:, ids[2]]  # 中段
        j4 = jp[:, ids[3]]  # 指尖
        ratio = j4 / (j3 + 1e-6)  # (N,)
        excess = torch.clamp(ratio - ratio_target, min=0.0)  # 超标量
        scores.append(torch.exp(-excess / 0.5))  # 比例内→1，超标渐变衰减
    raw = torch.stack(scores, dim=-1).mean(dim=-1)  # (N,)
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

    # 1) 五指贴面（touch 均值）→ 0~1
    dists = _fingertip_cube_surface_dists(env)  # (N,5)
    touch = (1.0 - torch.tanh(dists / touch_std)).mean(dim=-1)  # (N,)

    # 2) 五指有力（force 均值）→ 0~1
    forces = fingertip_contact_force(env)  # (N,5)
    force = (1.0 - torch.exp(-forces / force_std)).mean(dim=-1)  # (N,)

    # 3) 对向夹持（grip）→ 0~1：cube 局部 y 方向对向力（与 opposition_reward 同算法，不门控）
    from isaaclab.utils.math import quat_apply_inverse
    fvec = _fingertip_force_vectors(env)  # (N,5,3) 世界系
    cube_quat = env.scene["cube_obj"].data.root_quat_w  # (N,4)
    quat_rep = cube_quat.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
    fy = quat_apply_inverse(quat_rep, fvec.reshape(-1, 3)).reshape(fvec.shape)[..., 1]
    F_py = torch.relu(fy).sum(dim=-1)
    F_ny = torch.relu(-fy).sum(dim=-1)
    grip = torch.tanh(torch.minimum(F_py, F_ny) / grip_scale)  # (N,) 0~1

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
    """(v96) 惩罚手指 12 维动作幅度——逼手指输出 0（静止）。

    hand_action_penalty 只罚"弯曲量"（一阶矩），手指仍可在"不弯"的前提下抖动（高熵）。
    本项直接罚策略输出平方（二阶矩），唯一低惩罚解 = 手指输出 0（确定性静止），
    给无目标的手指 12 维提供确定性约束，根治 entropy_coef 熵奖励下的手指熵漂移/熵爆。
    动作拼接顺序：arm_action(6D OSC) 在前，hand_action(12D GroupedHandAction) 在后。
    Stage 2 激活 finger_contact 时需调低本项（让手指能动抓取）。

    [2026-09-03] 加 TCP 距离门控 gate_dist：抓取奖励（finger_close gate_std=0.03 等）在
      TCP 距 Cube > gate_dist 时≈0 → 手指 12 维无梯度 → 熵漂移 → 熵爆（本次 entropy 10.3）。
      门控后：远→1 罚（逼手指 0 防熵漂移），近→0 放开（抓取阶段手指自由弯曲）。
      （与 hand_action_penalty 的 gate_dist 同语义：远罚近放）
    """
    action = env.action_manager.action  # (N, 18)
    hand = action[:, 6:]  # 手指 12 维
    val = torch.clamp(torch.sum(torch.square(hand), dim=-1), max=10.0)
    if gate_dist is not None:
        # 距离门控：远→1（罚动作幅度逼 0），近→0（放开给抓取）
        d = _tcp_cube_dist(env)
        val = val * (d > gate_dist).float()
    return val


def hand_action_penalty(
        env: ManagerBasedRLEnv,
        gate_dist: float | None = None,
) -> torch.Tensor:
    """(v77) 惩罚手指关节弯曲量——Stage 1 手指保持伸直。

    play(model_400)：hand 碰 cube 时拇指指根被顶弯向掌心，回升后保持弯——
    旧版只罚动作（action=0 保持当前姿态），手指被碰弯后没有"回伸直"梯度。
    改为罚 max|finger joint_pos|（离伸直位 0 越远越罚），被碰弯后也会回到伸直。
    ⚠️ Stage 2 激活手指奖励时必须 weight→0（否则与抓取弯曲冲突）。

    v91: 改为相对初始姿态——原 max|joint_pos| 把拇指 init 的 ±0.3（故意避开限位）当弯曲
    常数惩罚（每步 -0.15），且诱导策略把拇指弯向 0 撞限位。改为 max(|joint_pos| − |init|)：
    初始弯曲不计入，只有被真正碰弯（超过初始）才罚；比初始更伸直（clamp 到 0）不罚。
    基准用 default_joint_pos（= init_state.joint_pos，单一来源）。

    [用户] gate_dist：加 TCP 距离门控——只有 TCP 距 Cube > gate_dist 才罚弯曲
      （接近阶段强制伸直，到位后放开自由抓取，不干扰抓取）。
      治 play 发现的"手指提前弯曲"（未接近就弯，小指/无名指连锁通道先弯）：
      门控奖励只控制给分、不控制动作，策略提前弯没惩罚 → 走捷径。
    """
    robot: Articulation = env.scene["robot"]
    ids_list = _get_finger_joint_ids(robot)
    all_ids = torch.cat([t for t in ids_list if t.numel() > 0]).to(robot.device)
    # 初始关节位置（init_state 设定，如拇指 ±0.3 避开限位）——相对它度量弯曲
    # v92: default_joint_pos 兼容 1D (num_joints,) 与 2D (num_envs, num_joints)（部分版本）
    dj = robot.data.default_joint_pos
    init_abs = (dj[0, all_ids] if dj.dim() > 1 else dj[all_ids]).abs()  # (F,)
    bend = torch.abs(robot.data.joint_pos[:, all_ids]) - init_abs  # (N, F) 超出初始的弯曲量（可为负=更伸直）
    val = torch.clamp(bend, min=0.0).max(dim=-1).values  # (N,) 最弯的那根超出的部分
    if gate_dist is not None:
        # [用户] 距离门控：远→1（罚弯曲），近→0（放开）
        d = _tcp_cube_dist(env)
        val = val * (d > gate_dist).float()
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
    v = robot.data.body_link_vel_w[:, tcp_body_id, :3]  # (N, 3) 世界坐标系线速度（link frame）
    speed = torch.norm(v, dim=-1)  # (N,)
    pen = (speed / vel_std) ** 2  # (N,) 平方惩罚高速
    if gate_dist is not None:
        d = _tcp_cube_dist(env)
        gate = 1.0 - torch.tanh(d / gate_dist)  # 近→1，远→0 平滑门控
        pen = pen * gate
    # v97: clamp max=100——物理瞬态时 speed 爆表（数 m/s）→ pen=(speed/0.05)² 可到数万，
    #   单步 reward -0.05×数万 污染 value（value loss 149036 崩溃的直接元凶）。
    #   正常接近 speed<0.3m/s（pen<36），clamp 100 只截断物理爆炸，不干扰正常训练。
    return torch.clamp(pen, max=100.0)


def palm_press_penalty(
        env: ManagerBasedRLEnv,
        clearance: float = 0.005,  # 允许手掌低于 cube 顶面的最小间隙（5mm，轻触级别）
        pen_std: float = 0.015,  # 压入深度归一化（压入 1.5cm → tanh(1)≈0.76）
        xy_gate: float = 0.04,  # 手掌水平投影距 cube 质心的门控（6cm cube 半边长 3cm + 1cm 余量）
) -> torch.Tensor:
    """(v74) 惩罚手掌压穿 cube——治"直接撞上去"。

    用户 2026-08-27 确认：TCP 离手掌 5cm，cube 完全可以在下方（不是 TCP 位置问题）。
    撞的根因：reach/success 只度量 TCP 距离、无方向约束，策略发现"用手掌直接推 cube"
    就能让 TCP 距离变小拿满分，而手掌本体先撞上 cube。
    本项只罚"手掌水平投影在 cube 内 (xy_dist<xy_gate) 且手掌 z 低于 cube 顶面-clearance"，
    压入越深越重（tanh 饱和）。正确悬停（手掌在 cube 上方）不触发，不影响正常接近。
    """
    palm_pos = _get_palm_pos(env)  # (N,3)
    cube_pos = _get_cube_pos(env)  # (N,3)
    xy_dist = torch.norm(palm_pos[:, :2] - cube_pos[:, :2], dim=-1)  # (N,)
    cube_top = cube_pos[:, 2] + _CUBE_HALF_SIZE  # (N,) cube 顶面 z
    depth = cube_top - clearance - palm_pos[:, 2]  # (N,) >0 = 压入量
    press = torch.tanh(torch.clamp(depth, min=0.0) / pen_std)  # (N,)
    over_cube = (xy_dist < xy_gate).float()  # (N,)
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
    tcp = _get_tcp_pos(env)  # (N,3)
    cube = _get_cube_pos(env)  # (N,3)
    xy_dist = torch.norm(tcp[:, :2] - cube[:, :2], dim=-1)  # (N,)
    depth = cube[:, 2] - tcp[:, 2]  # (N,) >0 = TCP 低于质心
    pen = torch.clamp(depth - min_dist, min=0.0) / pen_std  # (N,)
    return (xy_dist < xy_gate).float() * pen  # (N,)


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
    q_curr = robot.data.body_link_quat_w[:, tcp_body_id]  # (N, 4) 世界坐标系

    # 缓存初始姿态
    if not hasattr(env, "_tcp_init_quat_w"):
        env._tcp_init_quat_w = q_curr.clone()
    q_init = env._tcp_init_quat_w  # (N, 4)

    # |q_curr · q_init| — 四元数点积的绝对值
    dot = (q_curr * q_init).sum(dim=-1).abs()  # (N,)
    dot = torch.clamp(dot, -1.0, 1.0)
    ang = 2.0 * torch.acos(dot)  # (N,) 0~π

    # v97: clamp max=100——平方无上界（ang=π 时 pen≈438），探索期手腕瞬态大偏离会污染 value。
    #   正常偏离<40°（pen<30）、探索 60°（pen~160）→ clamp 100 保留正常+探索惩罚，只截断极端。
    return torch.clamp((ang / ang_std) ** 2, max=100.0)


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


def grip_force_reward(
    env: ManagerBasedRLEnv,
    opp_thresh: float = 0.12,
    force_std: float = 0.15,
    stab_std: float = 0.15,
) -> torch.Tensor:
    """(v50→v98) 超额对向力奖励——grip(tanh) 饱和后的持续加压梯度。

    grip 用 tanh(opp/scale)，opp 接近 scale 时梯度趋零（饱和），策略停在“收益递减点”加压不动。
    本项给 opp 超过 opp_thresh 的“超额”部分 1-exp 奖励：在大力范围梯度指数衰减但不归零，
    逼策略持续加压。三点设计：
      - 增量式：opp < opp_thresh 时 reward=0（不改变现状 grip 分数，不掉分 → 不触发熵漂移）
      - 只认 y 向对向力 opp（不认下压/侧推）→ 不破坏抓取姿势（不奖励下压作弊/侧摆挤压）
      - EMA 平滑 raw opp 滤抖（连续力信号的历史教训）

    [2026-09-08 回退 gate] 曾加 TCP 距离门控堵 play(12400) 发现的“手掌远离 + 四指勾棱边”虚高 opp，
      但加 gate 后策略被迫重学“贴住夹紧”（物理上手掌太近→手指碰 cube 顶面），entropy 升更快、
      奖励掉、波动大 → 失败回退。结论：逼力阶段只逼力（不堵姿势），勾棱边问题后置到 lift 阶段
      （cube 必须真的离桌 → 勾棱边夹不起 → 策略被迫真夹紧）。
    """
    opp = _opposition_force(env)                              # (N,) raw 对向力
    if not hasattr(env, "_grip_force_ema") or env._grip_force_ema.shape[0] != env.num_envs:
        env._grip_force_ema = opp.clone()
    ema = env._grip_force_ema
    ema = 0.3 * opp + 0.7 * ema
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf == 0
        if torch.any(reset_mask):
            ema[reset_mask] = 0.0
    env._grip_force_ema = ema
    excess = torch.clamp(ema - opp_thresh, min=0.0)
    raw = 1.0 - torch.exp(-excess / force_std)
    # [2026-09-09 稳定门控] cube 非目标运动（xy 滑动 + 翻滚）越大，加力收益越低。
    #   治"推-追循环"（加力→推 cube→掉稳定分→减力→追回）。与 08-08 回退的 TCP 距离门控不同：
    #   那个堵"姿势"，这个堵"加力破坏稳定"（真实功能问题）。只盯 xy+角速度（z 上升=被夹起=目标运动，不罚）。
    cube = env.scene["cube_obj"]
    slide = cube.data.root_lin_vel_w[:, :2].norm(dim=-1)      # (N,) xy 滑动速度
    tumble = cube.data.root_ang_vel_w.norm(dim=-1)            # (N,) 翻滚角速度
    non_target = slide + 0.1 * tumble                         # 非目标运动综合
    stability = torch.exp(-non_target / stab_std)             # (N,) 1=稳，→0=动
    return stability * raw


def _fingertip_force_vectors(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(W) 五个指尖接触力向量 → (N,5,3)，单位 N。force_matrix_w[:,0,0,:] = 世界系力向量。"""
    vecs = torch.zeros(env.num_envs, 5, 3, device=env.device)
    sensor_names = ["contact_thumb", "contact_index", "contact_middle", "contact_ring", "contact_little"]
    for i, name in enumerate(sensor_names):
        sensor = env.scene.sensors[name]
        fmat = sensor.data.force_matrix_w  # (N, B, M, 3)
        if fmat is not None and fmat.numel() > 0:
            vecs[:, i] = fmat[:, 0, 0, :]
    return vecs


def _opposition_components(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,) 两侧对向力分量 [F_+y, F_-y]（cube 局部 y 两侧力分别求和，未取 min，raw 未平滑）。

    供 _opposition_force（取 min）与观测函数（opposition_force_positive/negative）复用。
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


def opposition_force_positive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(观测用，weight=0) +y 侧对向力 F_+y，供 tensorboard 观察两侧是否平衡。"""
    F_py, _ = _opposition_components(env)
    return F_py


def opposition_force_negative(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(观测用，weight=0) -y 侧对向力 F_-y，供 tensorboard 观察两侧是否平衡。"""
    _, F_ny = _opposition_components(env)
    return F_ny


def opposition_reward(
        env: ManagerBasedRLEnv,
        gate_dist: float = 0.12,
        scale: float = 1.0,
        d_std: float = 0.01,
) -> torch.Tensor:
    """(v52) 对向夹持奖励（力封闭的平滑形式）——专治"四指挤同侧、无法夹"。

    背景（v51 诊断）：四指全压 +x 面 → 所有力同向 → 无对向 → cube 侧滑。
    grip_force（总力）会被单侧猛压骗到高分，无法区分"抓得稳"与"单侧推"。

    [2026-09-04] 世界系 → cube 局部系，只取局部 y 轴对向：
      抓握方向 = "手轴→四指方向" = cube 局部 y，拇指/四指沿 y 对向夹持；
      x 方向无手指抓握（侧摆方向），剔除 x 力——避免侧摆挤压的 x 向力被误计入 grip。
      对向度量：R = min(F_+y, F_-y)，再 tanh 饱和到 0~1。
    """
    opp = _opposition_force(env)  # (N,) 对向夹持力（cube 局部 y）
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
    opp = torch.tanh(ema / scale)  # 0→1 平滑饱和
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
      - 空间位置约束（指尖 z 坐标），非关节约束 → 避开手指 12 维锁死雷区。
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
        注：门控只防空抬；真正夹起还需 opp > mg/(2μ)=0.59N（μ=1.5，已确认手指 USD 无自带材质、全链路 fall back 全局），
        这由 cube_z 实际升高隐含要求。
      - 原 sqrt(lift) 在 0 附近导数→∞ 会放大噪声，改 tanh 导数有界。
    """
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["cube_obj"]
    close = torch.sigmoid(gate_steep * (gate_dist - _tcp_cube_dist(env)))
    flexed = torch.sigmoid(flex_steep * (_finger_flex(robot) - flex_thresh))
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
    tracking/bonus 反而给分 → 强化"乱动"→ 稳定抓取技能被覆盖（崩溃性遗忘）。
    clearance 只能测"cube 离没离桌"，不能测"是被提起来的还是被撞飞的" → 加手-物体距离门控：
      hand_gate = 1 − tanh(clamp(d − 6cm, min=0) / 3cm)
      正常悬停/抓握提起（d 1.5~5cm）→ 1.0 零影响；撞飞（d 10cm+）→ ≈0 堵死。
      物理语义："cube 离桌的功劳必须归于'手在 cube 附近'"。

    [与 command 解耦] 改高度目标只改 lift_std（不动观测，避免 OOD）。
    [翻滚防护] clearance = 最低顶点 z − 桌面（支撑函数），翻滚不触发。
    形状：r = tanh(clearance / lift_std) × hand_gate；lift_std=0.02（提起课程序）：
      抬 1cm 0.46 → 2cm 0.76 → 5cm 0.99（先"学会提起"，后期可拉回 0.06 学"提更高"）。
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
