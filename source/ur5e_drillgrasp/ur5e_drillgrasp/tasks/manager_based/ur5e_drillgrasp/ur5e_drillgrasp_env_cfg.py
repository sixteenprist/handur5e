# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from ur5e_drillgrasp.assets.robots.drill_ur5e import DRILL_UR5E_CFG
from ur5e_drillgrasp.assets.objects import DRILL_CFG
from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.sensors import ContactSensorCfg

from . import mdp


##
# Scene definition
##


@configclass
class Ur5eDrillgraspSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    num_envs = 2048
    env_spacing = 4.0
    # 新系统上 replicate_physics=True 导致 Cube 穿透桌子掉落（排查中）
    replicate_physics = True

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=None,
        ),

    )

    # ===== 桌面碰撞板（解决新系统复制物理下机器人USD内桌子碰撞失效）=====
    # 桌子在 drill_ur5e.usd 内，多 env 复制时其静态碰撞未生效 → Cube 穿透掉落
    # 每 env 加一块静态碰撞板兜底；顶面 = 真实桌面 0.75（2cm 厚 → 中心 0.74）
    table_collider: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableCollider",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.9, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),
        ),
    )

    # ===== 抓取目标物体（Cube）=====
    cube_obj: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,  # 保持 16（抓取稳定性需要，不动）
                solver_velocity_iteration_count=0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.5, 0.8)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.16, 0.175, 0.775),  # 桌面顶 0.75 + 半高 0.03
        ),
    )

    # 【后续换电钻时取消下面注释，注释掉上面】
    # object = DRILL_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")

    # robot
    robot = DRILL_UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # --- 指尖接触传感器（Stage 2 恢复：用户确认新系统传感器有数据）---
    # 配合 drill_ur5e.py 的 activate_contact_sensors=True；供 contact_force reward / fingertip_force 观测
    contact_thumb = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/hand/hand4/thumb4",  # thumb 在 hand4 下（其他手指仍在 hand 下）
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    contact_index = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/hand/hand4/index4",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    contact_middle = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/hand/hand4/middle4",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    contact_ring = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/hand/hand4/ring4",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    contact_little = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/hand/hand4/little4",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """
    动作配置类，用于定义机器人的手臂和手部动作参数
    包含手臂末端位姿控制和手部力矩控制的配置
    """
    # ===== 手臂动作：OSC（惯性解耦 + 重力补偿 + 笛卡尔阻抗）=====
    # [2026-09-17 防飞双保险] 原生 OSC 类没有任何出口限幅（_preprocess_actions 只×scale；
    #   cfg.clip 在该类里不生效——只有 IK 动作类会 clamp）→ 崩坏 σ 爆时手臂乱飞乱晃。
    #   换为本地安全子类（mdp/actions.py）：raw_clip 拦幅度 + max_delta 拦步间反转。
    #   正常训练零影响（只截极端；上线后按"正常阶段从不触发"核对）。
    #   对齐 SoftHand 的 max_drive_torque_delta / clip_actions 精神；调参：还晃→收紧、迟钝→放宽。
    arm_action = mdp.SafeOperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ],
        body_name="wrist_3_link",  # TCP 坐标系
        body_offset=mdp.OperationalSpaceControllerActionCfg.OffsetCfg(
            pos=(0.0, 0.08, 0.11),  # rewards/obs/reset 的 _BODY_OFFSET 已同步
            rot=(1.0, 0.0, 0.0, 0.0),  # base_link_1 局部坐标系 (B)，恒等四元数
        ),
        position_scale=0.005,  # ±5mm/步（速度上限 0.15m/s）[2026-09-14 用户决定 0.01→0.005]
        #   理由（对齐 SoftHand）：定位粒度 1cm→5mm（精确悬停/微调受益）；碰撞能量 -75%（防蹭跑、
        #   减接触颤动）；S2 微调更精细、S3 提起更稳（慢=稳）。
        #   ⚠️ 与 0.01 时代旧 ckpt 的动作映射不一致——旧 model_* 行为会变（同输出位移减半）；
        #   建议配合从头训练（若回滚续训，预期一段"重新适应期"）。
        orientation_scale=0.005,  # ±0.005rad 精细姿态（保持不变：护绕位速度 + 控漂移速率）
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            gravity_compensation=True,
            inertial_dynamics_decoupling=True,  # 惯性解耦，提升 OSC 稳定性
            motion_stiffness_task=(200.0, 200.0, 200.0, 194.0, 194.0, 194.0),
            # [2026-09-02] 位置 200 不动，姿态 180→194（参考 SoftHand UR5e）
            # [2026-09-02] 位置/姿态分开设阻尼：位置 1.5 治下沉，姿态 0.56 治侧翻（欠阻尼响应快，参考 SoftHand）
            motion_damping_ratio_task=(1.5, 1.5, 1.5, 0.3, 0.3, 0.3),
        ),
        # [2026-09-17 防飞] SafeOSC 子类参数：raw 幅度 ≤±2.0，步间变化 ≤1.5/步（先宽后窄）。
        raw_clip=None,
        max_delta=None,
    )
    # ===== 手指动作 =====
    hand_action = mdp.GroupedHandActionCfg(
        asset_name="robot",
        scale=1.0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """
        PolicyCfg类继承自ObsGroup，用于定义机器人观察空间的配置。
        该类包含了机器人、钻头、手掌和手指尖的各个观察项。
        """
        # ---------------- 机器人本体 ----------------
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, )
        last_action = ObsTerm(func=mdp.last_action, )

        # ---------------- TCP & 相对位置（世界坐标系）----------------
        # 去掉绝对位置 tcp_pos/cube_pos：与相对向量冗余（固定底座+固定 Cube 下无信息量）
        tcp_to_cube = ObsTerm(func=mdp.tcp_to_cube, )
        palm_to_cube = ObsTerm(func=mdp.palm_to_cube, )

        # ---------------- Cube ---------------
        cube_quat = ObsTerm(func=mdp.object_orientation, )

        # ---------------- 手掌 ----------------
        palm_pos = ObsTerm(func=mdp.palm_position, )
        palm_quat = ObsTerm(func=mdp.palm_orientation, )

        # ---------------- 指尖→Cube 相对位置（世界坐标系）----------------
        fingertip_to_cube = ObsTerm(func=mdp.fingertip_to_cube_surface, )

        # ---------------- 指尖触觉（接触力范数，5 维）----------------
        # [2026-09-15 重训版·闭环修复] 从 critic 移回 actor（98→103 维）：
        #   "始终贴面给力"是接触力维持任务，策略必须能感知力才能闭环（力掉了→加压）；
        #   只给 critic 只能"学得准"（优势质量），给不了"做得到"（策略无感知即无法调力）。
        #   S2 尚未出成果，弃旧 ckpt 重训成本最低——本版从头训练（所有旧 ckpt 不兼容）。
        #   （v71 曾因 109 维 ckpt 兼容把本项放 critic；现 ckpt 包袱已解除。）
        fingertip_force = ObsTerm(func=mdp.fingertip_contact_force, )

        # ---------------- 目标命令 ----------------
        # [2026-09-14 精简] 注释：固定目标下该命令是 7 维常量观测（无信息量），从头训练无兼容负担。
        #   未来做"目标随机化泛化"时再启用（policy 105→98 维，rsl_rl_ppo_cfg 注释已同步）。
        # goal_pose = ObsTerm(
        #     func=mdp.generated_commands,
        #     params={"command_name": "drill_pose"},
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 特权观测组（借鉴 SoftHand 的非对称 actor-critic）。

        映射关系见 agents/rsl_rl_ppo_cfg.py 的 obs_groups：
          actor ← policy 组（103 维；2026-09-15 指尖力 5 维从本组移回 actor）
          critic ← policy + critic 组（103 + 9 = 112 维；总维度守恒）

        只放"仿真里拿得到、真机/策略端拿不到或没必要拿"的真值信息，
        让 value 估计更准（success 依赖 cube 稳定性，而 actor 看不到它）。
        增删本组项不影响 actor 输入维度。
        """
        # ---------------- Cube 真值运动状态（3+3）----------------
        # success_stage1 要求 cube 线速度<0.10、角速度<0.20 才算"稳定悬停"，
        # actor 观测里没有任何速度项 → 给 critic，价值估计直接受益。
        cube_lin_vel = ObsTerm(func=mdp.object_lin_vel, )
        cube_ang_vel = ObsTerm(func=mdp.object_ang_vel, )

        # ---------------- Cube 绝对位置（3）----------------
        # 当前固定位置下近似常量；启用 reset_object 位置随机化（泛化阶段）后
        # 成为价值估计必需项，提前放入避免届时再动观测维度。
        cube_pos = ObsTerm(func=mdp.object_position, )

        # [2026-09-15] fingertip_force 已移入 PolicyCfg（actor 闭环修复）——
        #   本组不再重复放置（critic 经 policy 组自动包含该 5 维）。

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """
    事件配置类，用于定义和管理各种事件配置项。
    包含机器人重置和物体重置等相关事件的配置。
    """
    # ===== Domain Randomization（暂关闭，先跑通基础效果）=====
    # robot_physics_material = EventTerm(...)
    # robot_scale_mass = EventTerm(...)
    # object_physics_material = EventTerm(...)
    # object_scale_mass = EventTerm(...)
    # object_random_scale = EventTerm(...)
    # reset_object_pos = EventTerm(...)

    # ===== 重置与缓存 =====
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,  # 执行关节重置的功能函数
        mode="reset",  # 事件模式为重置
        params={  # 重置参数配置
            "asset_cfg": SceneEntityCfg("robot"),  # 指定重置目标为机器人实体
            "position_range": (-0.02, 0.02),  # 位置偏移范围
            "velocity_range": (0.0, 0.0),  # 速度偏移范围
        },
    )
    cache_tcp_orientation = EventTerm(
        func=mdp.cache_tcp_orientation_on_reset,
        mode="reset",
        params={},
    )
    reset_tcp_close_bonus = EventTerm(
        func=mdp.reset_tcp_close_granted,
        mode="reset",
        params={},
    )
    reset_goal_bonus = EventTerm(
        func=mdp.reset_goal_bonus_granted,
        mode="reset",
        params={},
    )

    # ===== 【泛化阶段启用】Cube 位置随机化（当前固定位置，先注释）=====
    # 注：Cube 边长 5cm，±3cm 总范围约 1.2 倍边长，够让策略依赖 cube_pos 自适应
    # reset_object = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": [-0.03, 0.03], "y": [-0.03, 0.03], "z": [-0.00, 0.00]},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("cube_obj", body_names=".*"),
    #     },
    # )

@configclass
class RewardsCfg:
    """Reward 权重配置（四指共享 / 分阶段，2026-09-14 精简版）。

    设计原则（参考 SoftHand）：一条任务链，相邻项职责严格错开、不重合不冲突；
    每阶段只留 2~3 个正奖励，权重错开（链尾任务项 > 链中过程项）：
      接近链（S1）: reach（远处拉近，dense std=1.0）→ success_reward（1.5cm 悬停+稳定，仅近场）
      姿态（S1）:   tcp_orientation（防翻转/乱转；SoftHand 同款从 -0.1 起步、逐步加大）
      抓取链（S2）: finger_reaching（表面距离引导 2.0）→ finger_contact（贴面力 3.5，拇指组+四指组两组聚合）
                    → contact_hold（接触占空比 2.0）→ grip（对向夹持力 3.0，链尾主导）
                    ——对应 SoftHand 的 reaching→contact→force 三档 + 本项对向力
      提起链（S3）: object_goal_tracking（离桌高度 dense 5.0）+ object_goal_bonus（到位一次性 1.0）
      惩罚: action_rate（平滑）+ hand_action_mag（S1 手指静止门控；S2 起归零）

    Stage 1 接近（已切出→S2，历史）: reach 3.0 + success_reward 4.0；惩罚 action_rate -0.05 + tcp_orientation -0.1
                            + hand_action_mag -0.2（手指静止防熵漂移，门控 0.05 保持）
    Stage 2 抓取（[2026-09-14 已激活]）: finger_reaching 2.0 + finger_contact 3.5 + grip 3.0（力封闭链尾）
                            + contact_hold 2.0 + fingertip_press -0.2；同时 hand_action_mag→0（放手）、
                            reach→2.0（防遗忘）、tcp_orientation→-0.05（松绑手腕绕位，历史预案）、
                            success→2.0（[2026-09-15 软着陆]：治"碰 cube ↔ 保悬停"转换期打架 → value 爆）
                            ✅ _opposition_force 已恢复（2026-09-14，grip/lift 共用）。
                            预案：grip 恒 0→开 thumb_opposition；hold 高而 grip≈0→查压顶作弊。
    Stage 3 提起（手动启用→改 weight）: object_goal_tracking→5.0 + object_goal_bonus→1.0
                            （保持 S2 的 grip/contact_hold；reach→0，不再需要接近）
    待命（weight=0）: tcp_velocity（撞飞）/ joint_vel（微项，SoftHand 开 -0.001 可参考）
                            / finger_close（与 finger_reaching 重合）/ thumb_opposition（拇指不绕位时开）
    历史：CurriculumCfg 已停用；lift_reward 已让位（注释保留）。

    [v2 重训 S1↔S2 切换清单 2026-09-15（力观测回归版·从头训）] 先 S1 达标再切 S2：
      S1: reach 3.0 | success 4.0 | hand_action_mag -0.2 | tcp_orientation -0.1 | S2 链全 0
      S2: reach 2.0 | success 2.0 | hand_action_mag 0 | tcp_orientation -0.05
          | finger_reaching 2.0 | finger_contact 3.5 | contact_hold 2.0 | grip 3.0 | fingertip_press -0.2
      （S2 链 = finger_reaching / finger_contact / contact_hold / grip / fingertip_press 五项）
      S2-α3（2026-09-16 最小增量）: reach 2.0 | success 2.0 | hand_action_mag -0.2 | tcp_orientation -0.05
          | finger_reaching 2.0 + thumb_opposition 2.0 + finger_contact 3.0；hold/grip 保持 0（贴住立住后梯次加）
      S2-α4（2026-09-16 近端精核）: 已回退（改值分布 → value 失配崩，迭代 1419）
      S2-α5（2026-09-16 "加零项"通道）：α3 基础上启用 finger_close 1.0（弯曲引导——破"手指不前进"死循环；
          历史预案原话：出现"伸直→不贴面→无梯度"死循环时恢复 1.0）
      S2-α6（2026-09-16 交棒）：thumb_opposition 2.0→0——play 实证"拇指 3/4 继续卷曲 → 对置分下降"
          （方向分与距离无关 → 把拇指锁在"悬空对置"局部最优）；撤掉反卷曲力，卷曲+贴面交棒
          close/contact，两侧"真用力"由后续 grip（对向 force min）验证
      S2-α7（2026-09-16 拇指专项）：新增 thumb_face_reach 0.5——"拇指尖→对侧面中心"紧核（σ=2.5cm），
          专补"最后 1cm"（GUI 手拖证实指尖可达对侧面=非结构死区；单指无 min，卷曲只加钱不扣分）
      （历史：α2 阶段1 = 只开 finger_contact 塑四指；"钉住"思路由 α3 直接三开实现）
      切换点（2026-09-15 更新，reach std 0.4 新核）：S1 reach≥2.4（≈距 cube 8cm）~2.6（≈5.4cm）
        且 success 开始出分（旧核 2.4 分=27cm 很松；新核 2.4 分含金量≈旧核 2.7+，勿按旧数值习惯早切）
    """

    # ══════════════════════ Stage 1：接近（已切出 → Stage 2，2026-09-14）══════════════════════
    reach: RewTerm = RewTerm(
        func=mdp.reach_reward,
        weight=2.0,  # [2026-09-14 S2 激活] 3.0→2.0 防遗忘（S1 历史值 3.0，能跑 ~13% success）
        # v14: std 0.4→1.0。旧值在 ~1m 处 tanh 饱和（奖励 0.01/步、梯度 0.16/m），
        # 远场拉不动 → 策略卡在 1m、noise std 膨胀到 0.52 不学习（v13@268 success=0）。
        # std=1.0：1m 处奖励 0.71/步、梯度 1.25/m（8×），0.5m→1.6，0.3m→2.1，0.1m→2.7（×3 权重）
        # [用户] std 0.45→1.0（回归 v14）：std 只改梯度随距离的分布，不移动悬停目标
        #   （悬停由 success 1cm 饱和型驱动）。当初收到 0.45 是为"贴面→悬停"额外梯度，
        #   现已被 success 1cm 阈值取代（贴面 d=2cm 拿 0 分）→ 放宽回 1.0 恢复远场梯度
        #   （1m 外不再饱和），加速接近收敛；近端 2cm→0 推力 0.13→0.06/步，由 success 覆盖
        # [用户] z_std 0.02：TCP 只允许在 cube 质心及以上（与 success 方向约束一致）
        # [2026-09-15 v2·40cm 场景] std 1.0→0.4（对齐 SoftHand object_tcp_distance 的 std=0.40）：
        #   初始距 40cm，"1m 远场饱和"顾虑不再存在；核收窄后中段（27→10cm）可得分差 0.49→1.03 分（×2.1）、
        #   近场梯度 3→7 分/m（推进边际 0.015→0.035 分/步）——治"过早收敛卡 2.2"。
        #   注：新核下分数含义变化（2.4 分≈距 8cm、2.6≈5.4cm）——切换标准已同步更新。
        params={"std": 0.5, "z_std": 0.02},
    )
    action_rate: RewTerm = RewTerm(
        func=mdp.action_rate_penalty,
        weight=-0.05,  # [2026-09-03] -0.02→-0.05：治抓取时手指颤动——只罚 a_t−a_{t-1} 变化量，
        #   不罚恒定弯曲（抓取允许稳定保持弯曲），只压"来回抖"的高频颤动。
    )
    # [v96 2026-09-01] 手指动作幅度惩罚——逼手指输出 0（静止），给无目标的手指通道提供确定性约束。
    #   根治：entropy_coef 熵奖励下手指通道无梯度 → 熵漂移 → 熵爆（v71:32.8 / 旧训练:29 / 本次:34.3）。
    #   对比（已删的旧 hand_action_penalty）：只罚弯曲量（一阶矩），手指仍可抖动（高熵）；本项罚动作平方（二阶矩）逼静止。
    # [v99 2026-09-01] -0.5→-0.2：-0.5 太硬，手指通道 σ 被压进负熵区（σ<0.242），总 entropy 30~40 iter 即掉负
    #   且持续减小，熵自校正被压制，手臂也快速塌到次优解（reach 卡 1.8~1.9，与 entropy_coef 0.001 时代同款过早收敛）。
    #   回到 -0.2：虽会慢熵漂移（300+ iter 到 11），但 reach 能爬到 2.4+，配合"reach 2.4 即切 Stage 2"可用。
    #   （根本解：只对手臂 6 维算 entropy bonus，让手指熵不进入 loss，见 rsl_rl PPO 子类化方案）
    hand_action_mag: RewTerm = RewTerm(
        func=mdp.hand_action_magnitude_penalty,
        weight=0.0,  # [2026-09-15 S2-α 恢复] 0→-0.2：手指空闲期防漂闸（S1 配方）——
        #   四指无任务时靠它压熵漂移；拇指绕位收益 2.0/步 >> 本项量级，不阻绕位动作。
        #   ⚠️ 状态切换项：S1/α 用 -0.2；S2 抓取期用 0.0（放手给抓取链）。
        params={"gate_dist": 0.06},
    )
    joint_vel: RewTerm = RewTerm(  # 关节速度 L2 惩罚
        func=mdp.joint_vel_l2,
        # [2026-09-14 精简] 待命：-0.001 量级过微，平滑已由 action_rate 负责。
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    tcp_velocity: RewTerm = RewTerm(
        func=mdp.tcp_velocity_penalty,
        # v15: 接近减速（原 0.0 关闭）—    # [2026-09-12 让位] lift weight 2.0→0：其"离桌越高越好"(tanh 单调) 与新 object_goal_tracking
    #   的"到达目标 0.79"方向矛盾（lift 激励继续抬，goal 激励停在 0.79）。夹持确认门控
    #   （close×flexed×gripped）已完整继承到 object_goal_tracking，lift 让位避免重复/矛盾。
    # lift: RewTerm = RewTerm(
    #     func=mdp.lift_reward,
    #     weight=0.0,
    #     params={"gate_dist": 0.05, "gate_steep": 10.0, "flex_thresh": 0.2, "flex_steep": 5.0,
    #             "z_thresh": 0.78, "lift_std": 0.01, "lift_force_thresh": 0.03, "force_steep": 100.0},
    # )—距 cube<15cm 后平滑减速到 ~6cm/s，
        # 把"直接撞上"变成柔和接触，减少 cube 位移（也让 success 的"稳定"更易满足）。
        # 权重不能太重：否则最后 10cm 走太慢，299 步 episode 内到不了成功区。
        # [加强] -0.002→-0.05、vel_std 0.1→0.05：0.3m/s→1.08/步、0.5m/s→3.0/步，
        #   压住"接近速度过快"（用户 1500 轮观察）
        # [2026-09-14 精简] 待命：Stage 1 若仍出现"高速撞飞 cube"再恢复 -0.05。
        weight=0.0,
        params={"vel_std": 0.05, "gate_dist": 0.15},
    )
    tcp_orientation: RewTerm = RewTerm(
        func=mdp.tcp_orientation_penalty,
        # [2026-09-14 对照 SoftHand] 激活：其 tcp_orientation_stability 从 -0.1 起步、
        #   随训练逐步加大（-0.3→-5）贯穿全程，防翻转/乱转。
        # [同日 漂移修复 v1] -0.05→-0.2：治 5100 轮姿态积分漂移（OSC pose_rel 恒定偏置积分）。
        # [同日 崩坏回退] -0.2→-0.1：从头训练下 -0.2 使手腕探索被重罚 → 全维 σ 450 轮即崩到
        #   entropy -28（焊死后无纠错力，1100 轮姿态逃逸崩坏）。回到 -0.1（SoftHand 原始起步值），
        #   **分档计划**：训练 >2000 轮且姿态稳定后手动加到 -0.15/-0.2（盯 entropy 不低于 -10）。
        #   配套：rewards.py 内 clamp 30（单步上限 ≤3@-0.1，柔和）。
        #   ⚠️ Stage 2 注意：手腕绕位旋转与 grip 有历史冲突（2026-09-03），届时可回调 -0.05。
        weight=-0.20,  # [2026-09-15 S2-α 激活] -0.1→-0.05：松绑手腕旋转（拇指绕位需手腕轻转配合）
        params={"ang_std": 0.25},
    )
    tcp_close_bonus: RewTerm = RewTerm(
        func=mdp.tcp_close_bonus_once,
        weight=0.0,  # 一次性奖励：TCP 首次贴近 Cube
        # v21: 4.5cm→4cm——v20 偏宽；4cm = 距 6cm cube 表面 1cm，真正轻触
        # [用户] bonus 8.0→5.0（撤回）：8.0 一次性尖峰在策略大量穿越 4cm 边界时密集发放，
        #   → 稀疏大奖励 → value loss 爆 28.8（952→1052 iter）→ 策略退化。回 5.0 控尖峰。
        params={"distance_threshold": 0.040, "bonus_value": 2.0},
    )
    # object_distance_penalty: RewTerm = RewTerm(
    #     func=mdp.object_distance_penalty,
    # 密集引导：touch 是距离连续梯度（贴面有 base=0.4 底分），force/grip 做稀疏加分。
    # [Stage 2 2026-09-01] finger_close 关节角弯曲奖励：给"弯曲量"直接密集梯度，
    #   打破"伸直→不贴面→无梯度→伸直"的 bootstrap 死循环（finger_contact 只在贴面给分太稀疏）。
    # [2026-09-02] 硬门控→软门控 + 门控 8cm→3cm（gate_std=0.03）+ weight 1.5→0.5：
    #   finger_contact 已 0.49 起来，finger_close 退居二线，减弱过程导向副作用
    #   （抬手腕空弯/关节3过弯/抓取时颤动），真悬停才奖励弯曲。
    finger_close: RewTerm = RewTerm(
        func=mdp.finger_close_reward,
        # [2026-09-14 精简] 待命：与 finger_reaching（贴面力）功能重叠；
        #   若 S2 出现"伸直→不贴面→无梯度"的 bootstrap 死循环，恢复 1.0。
        # [2026-09-16 恢复 1.0] 拇指弯曲梯度（治 play 所见"拇指3/4 不弯、不触面"）：四指已会弯 → cross min 被拇指钉住，
        #   梯度≈6:1 几乎全落拇指（关节3=1.5 / 关节4=0.8 权重最高，正好是弯不够的那两节）。
        weight=0.0,  # [2026-09-05 历史] 1.0→0.5；[2026-09-16] 恢复 1.0
        # [2026-09-04] gate_std 0.03→0.08：弯曲应早于接触（时序：接近15cm→弯曲8cm→接触5cm→力封闭1.5cm）；
        #   max_flex 0.5→0.6：给"贴面需要更多弯曲"留余量（有层级门控+纯力判据约束，不会过度蜷缩）。
        # [2026-09-17 梯次修正] gate_std 0.05→0.08——恢复"塑形先于接触"（0.05 时 7-9cm 门开度仅
        #   0.05~0.12 ≈ 无付款，时序名存实亡）。全链门控顺序（d=手-物距离）：
        #   reach 常开 → reaching 9→5cm（指尖就位）→ close 8cm 起明显（手指塑形/卷曲，本项，悬停区渐强）
        #   → contact 7→5cm+触力（贴面给力）→ grip 2.5→1.5cm（力封闭，待启用）→ success 1.5cm（悬停稳定）。
        #   σ=0.08 开度：10cm 0.15 / 8cm 0.24 / 5cm 0.45 / 3cm 0.64 / 2cm 0.76（10cm 处已 0.15 且继续
        #   衰减，防"空中空弯"）。
        # [2026-09-17 贴近因子] prox_min=0.25：close 从"卷就给钱"改为"卷且贴才给全额"——
        #   factor = 0.25 + 0.75×指尖贴近度（与 reaching 同口径）。浅卷+远(prox≈0.2)→×0.40
        #   （年金 0.18→~0.07）；卷着凑近(0.5)→×0.63；贴面(≈0.7)→×0.78（有效封顶随之 ×0.78）。
        #   动机：close（卷）/reaching（近）可分离 = "浅卷+不接近"死锁 → 耦合成"卷着靠近"双收入。
        #   回退：prox_min=1.0（一键回旧行为）。
        params={"gate_std": 0.03, "max_flex": 0.60, "prox_min": 0.25},
    )

    # ══════════════ Stage 2：抓取（[2026-09-14 已激活] 链：reaching 2.0 → contact 3.5 → grip 3.0 → hold 2.0，
    #   press -0.2 防守；reach→2.0、hand_action_mag→0、tcp_orientation→-0.05。
    #   ✅ _opposition_force 已恢复。预案：grip 恒 0→thumb_opposition；hold高+grip≈0→查压顶）══════════════
    finger_reaching: RewTerm = RewTerm(
        func=mdp.finger_reaching_reward,
        weight=3.0,  # [2026-09-16 S2-α3 恢复] 0→2.0：最小增量三开之一（接近引导=四指悬垂手型的基础）
        # [2026-09-14 修正] params 与函数签名对齐：本机版是纯"表面距离"引导（touch_std 版）；
        #   原 params（force_std…）是力版旧键，S2 启用时会 TypeError 崩溃。
        #   距离引导负责 bootstrap，力度档由 finger_contact 承担。
        # [2026-09-15 SoftHand 对齐] touch_std 0.03→0.10（宽核）：配合 rewards.py 改"纯质心距离"——
        #   中远场连续梯度（表面 5cm≈0.34，旧核 0.003 死区）+ 近场无提前饱和（贴面 0.71 渐进）。
        # [2026-09-16] S2-α4 近端精核（near_*）已回退——改值分布 → value 失配崩。
        params={"touch_std": 0.08, "dist_threshold": 0.03, "d_std": 0.02},
    )
    # [2026-09-14 贴合链补档] 指尖贴面力（连续 tanh(f/σ) + EMA 平滑；聚合 0.8·mean+0.2·min 软逼全指）——
    #   分工：finger_reaching 管"靠近"→ 本项管"贴住且有力"→ contact_hold 管"别断"→ grip 管"对向夹住"。
    #   [2026-09-07 失稳回退] σ 曾 0.10→value loss 0.004→0.72 尖峰（0.1N 工作点 grad=4.2 放大噪声 3 倍）；
    #   σ=0.06 时 0.1N 处 grad=1.28（饱和区，抖动被压死）；weight 4.0→3.5 补偿 tanh 值回升。
    #   ⚠️ 若 value loss 仍 >0.1 或尖峰再现，立即回退二值版（git 历史）。
    finger_contact: RewTerm = RewTerm(
        func=mdp.finger_contact_reward,
        weight=0.0,  # [2026-09-15 S2-α2 阶段1] 0→3.0：四指接触塑形——先让四指贴住 cube 弯下
        #   （四指被"钉"在 cube 上后，阶段 2 再开拇指对置——此时对置判据不可用"摊平手"作弊）。
        params={"force_std": 0.06, "dist_threshold": 0.03, "d_std": 0.02},
    )
    # [2026-09-13 颤动修复] 接触持续性（占空比）奖励——用户 play 观察：指尖贴面"有时接触
    #   有时不接触、变化很快、肉眼可见"（contact chatter；cube 未被推飞）。
    #   现有项对断续"盲"：finger_contact EMA α=0.15 / grip EMA α=0.3 + 时间平均读数把
    #   秒级断续抹平（1.2N×50% + 0×50% ≈ 0.6N×100%）→ 策略无消除颤动梯度。
    #   本项：瞬时二值接触事件（力>0.03N）+ 短窗 EMA 占空比 → 对断续敏感（10Hz 级可捕捉）、
    #   对力噪声鲁棒。分工：finger_contact 管"力多大"、本项管"别断"（持续贴合）。
    #   weight 2.0 不主导；零成本挂上（接触前恒 0，学会接触后渐进爬升，无尖峰）。
    contact_hold: RewTerm = RewTerm(
        func=mdp.contact_persistence_reward,
        weight=0.0,  # [2026-09-15 S2-α 暂关] 2.0→0（β 阶段恢复）
        params={"force_thresh": 0.03, "dist_threshold": 0.05, "d_std": 0.02},
    )
    # 正确抓取 = 力封闭（对向两面力取 min）。[Stage 2 2026-09-02] 0.0→1.0：激活力封闭奖励。
    # [2026-09-03] 1.0→3.0：play 发现力封闭需要"拇指绕到四指对侧"的手腕旋转，被 tcp_orientation
    #   惩罚压制（转 30° 罚 -0.61/步 vs grip 上限 +1/步）。grip ×3 让力封闭收益买得起姿态代价。
    # [2026-09-03 绕位阶段] 3.0→0.0：暂时清零（绕位阶段不需要对向力确认），抓握阶段恢复 3.0。

    grip: RewTerm = RewTerm(
        func=mdp.opposition_reward,
        weight=0.0,  # [2026-09-15 S2-α 暂关] 3.0→0：力封闭需"拇指已绕位"前置（β 阶段恢复）
        # [2026-09-12 回退] 曾 5.0→4.0 + 内部加 balance/align 门控（完整力封闭），
        #   但最大单项结构大改 → value 失配 → 续训一开始 reward 全崩（奖励归零只剩惩罚）。回退纯 tanh(min)。
        #   [2026-09-07] 3.0→5.0：力起不来，提权重让对向力成为最大单项；只加不砍。
        params={"gate_dist": 0.015, "scale": 0.10, "d_std": 0.01},
    )


    # [2026-09-03] 拇指对侧奖励：引导拇指绕到四指对侧（力封闭的几何前提）。
    #   grip 是"力的稀疏确认"（拇指绕到位前恒 0，无梯度），本项给"几何密集引导"
    #   （接触前就有梯度）——互补：本项把拇指引到位，grip 确认夹住。
    thumb_opposition: RewTerm = RewTerm(
        func=mdp.thumb_opposition_reward,
        # [2026-09-16 S2-α3 恢复] 0→2.0：最小增量三开之一（对置引导——四指在位后拇指引到对面）
        # [2026-09-16 S2-α6 交棒] 2.0→0：play 实证"拇指 3/4 继续卷曲 → 对置分下降"——方向分与距离无关，
        #   把拇指锁在"悬空对置"局部最优（伸直保分、卷曲掉分），与"卷曲贴面"直接打架，阻断最后一厘米。
        #   绕位使命已完成（稳定 1.7~1.8，行为已进权重），交棒 finger_close（卷曲）+ finger_contact（贴面）。
        #   反向梯度清除后，close 的卷曲分（min 结构被拇指钉住）才推得动 3/4。grip 段自带"两侧 min"，如
        #   需方向锚届时再临时恢复。
        weight=0.0,
        params={"gate_std": 0.05},
    )
    # [2026-09-16 S2-α7 拇指专项] "拇指尖 → 对侧面中心"紧核接近奖励：专补"最后 1cm"。
    #   实证链：α6 交棒后 play 仍见"拇指 3/4 不卷、不贴面"；GUI 手拖四个拇指关节证实
    #   指尖可达对侧面（非结构死区）→ 缺"末段密集付款"。σ=2.5cm 只管最后一小段；
    #   目标点=四指对侧面中心（门把手式，摸顶/邻侧无分）；TCP 软门控=悬停区才计分。
    #   单指、无 min：不会先补四指，也不会对卷曲反付钱。跑基线：weight 置 0 即可。
    thumb_face_reach: RewTerm = RewTerm(
        func=mdp.thumb_face_reach_reward,
        weight=1.0,  # [2026-09-18 R1 对称目标对] 0→1.0：与 four_face_reach 成对启用——
        #   "接近"统一为两侧独立目标点距离付款（无 min、无方向分、不对卷曲罚分），拆"跷跷板"根因。
        #   历史 09-16 曾配 σ×2 启用后搁置（当时探索已死 + opp 反向拉力在场）；现两前提已消
        #   （σ 地板 0.12 / opp→0）→ 按 R1 重启用。回退：weight 置 0。
        params={"sigma": 0.05, "gate_std": 0.08},
    )
    # [2026-09-18 R1 对称目标对·四指端] 工作对（中指+无名指）指尖 → 同侧面中心；与拇指端同构、同权。
    #   两侧合起来 = 对称目标对：每侧一条到"面上明确点"的单调付款，互不卡付。
    #   ⚠️ 依赖 rewards.py 的 four_face_reach_reward（2026-09-17 版仍在）；回退：weight 置 0。
    four_face_reach: RewTerm = RewTerm(
        func=mdp.four_face_reach_reward,
        weight=1.0,
        params={"sigma": 0.05, "gate_std": 0.08},
    )
    # [v69][SoftHand 借鉴] 指尖合力向下分量惩罚——防"手指把 cube 压向桌面"的下压作弊。
    #   世界系 Z：向下合力 >0.02N 才计罚；水平对向夹持（力封闭）Z 分量小 → 少罚。
    # [堵洞 2026-09-01] 小指/无名指弯下来压 cube 提高 success（隐性收益 +3.0/步 > hand_action -0.24/步），
    #   原 gate_dist 1cm 太严（悬停时 TCP 距 cube ~1.5cm，门控恒不满足 → 惩罚从未触发，日志恒 0）。
    #   放宽 gate_dist 1cm→4cm 覆盖整个悬停区；weight -0.1→-0.5 让"手指压 cube"变成亏本买卖。
    fingertip_press: RewTerm = RewTerm(
        func=mdp.fingertip_press_penalty,
        weight=-0.2,  # [2026-09-14 S2 激活] 0→-0.2：防"手指压 cube 提 success"作弊（防守项）
        # [2026-09-03 历史] -0.5→-0.2：单步惩罚上限降至 -0.2，降接触瞬态尖峰对 value 的冲击。
        params={"force_std": 1.0, "deadzone": 0.02, "gate_dist": 0.04},
    )


    # ===== 提起项（Stage 3 启用；lift_reward 已让位，注释保留）=====

    # [2026-09-13 方案一+撞飞修复] 提起奖励 —— "离桌高度"版 + 手-物体门控：
    #   ① 结构：r = tanh(clearance / lift_std) × hand_gate。
    #   ② 撞飞 hack 修复（崩溃证据）：机械臂乱动把 hand 带离 cube、cube 被撞飞 → clearance>0
    #      → tracking 反而给分 → 强化"乱动"→ 崩溃性遗忘。hand_gate = 1−tanh(clamp(d−6cm,0)/3cm)：
    #      正常悬停/抓握（d 1.5~5cm）→ 1.0 零影响；撞飞（d 10cm+）→ ≈0 堵死。
    #   ③ lift_std=0.02（提起课程序）：抬 1cm 0.46 → 2cm 0.76 → 5cm 0.99（先"学会提起"，
    #      稳固后可按 0.02→0.03→0.06 反向课程学"提更高"）。
    #   ④ weight 5.0：与 success(4.0) 同级、比 grip(3.0) 大（驱动举起）、小于夹持总和 11.0。
    # ══════════════ Stage 3：提起（切阶段时启用：object_goal_tracking→5.0 / object_goal_bonus→1.0，
    #   保持 S2 的 grip/contact_hold；reach→0）══════════════
    object_goal_tracking: RewTerm = RewTerm(
        func=mdp.object_goal_tracking_reward,
        weight=0.0,  # [2026-09-14 精简] S3 启用（→5.0）
        params={"lift_std": 0.02, "table_height": 0.75, "hand_gate_dist": 0.06, "hand_gate_std": 0.03},
    )
    # [2026-09-13 方案一+撞飞修复] 提起达标一次性奖励：
    #   判定 clearance > 0.15（离桌 15cm）**且手在 cube 8cm 内** → 一次性 bonus=10.0。
    #   撞飞时手被带离（d 大）→ 不触发（旧版撞飞腾空>15cm 会误发 10 分 → 强化乱动）。
    object_goal_bonus: RewTerm = RewTerm(
        func=mdp.object_goal_bonus_once,
        # [2026-09-14 精简] S3 启用（→1.0）；bonus=10 属一次性尖峰，若 value 冲击大再降到 5.0。
        weight=0.0,
        params={"clearance_threshold": 0.15, "bonus": 10.0, "table_height": 0.75, "hand_dist_max": 0.08},
    )
    # ===== 成功（观测模式：纯 Stage 1，只要求手掌贴近 + 稳定，不要求手指/抬起）=====
    # 逐步发放、不终止；权重 30→3：保留每步"拉近+保持"梯度，双峰回报方差降 10×（9000→900）
    # 防 entropy 暴涨崩溃（v5/v9/v10 权重 30 时熵 16+ 即崩盘前兆）
    success_reward: RewTerm = RewTerm(
        func=mdp.success_stage1_reward,
        weight=2.0,  # [2026-09-15 S2 软着陆] 4.0→2.0：治"悬停交棒悬崖"——
                     #   "碰 cube"（手指链引力）与"保 cube 稳定"（success）物理对立：碰了 cube 一晃，
                     #   success 从 3.5/步直落 0（橙色 run 2500-2600 实录：接触建立与 value 1.98 爆同步）。
                     #   两个奖励在转换期打架振荡 → value 地震。降权 = 减小赌注，让 S2 转换软着陆。
                     #   若仍见 success↔contact 交替振荡，下一步升级为"交棒门控"（接触建立后 success 淡出）。
        # v21: 7.5cm→6.5cm——v20 太宽；6.5cm 表示真正包住（距 7cm cube 表面 3cm）
        # v23: 稳定阈值放宽 lin 0.05→0.10、ang 0.10→0.20——抓握必然推动 cube，
        # 严的"稳定"在惩罚"碰 cube"（碰就掉 3/步），是"手指不接触"的结构性根源
        # [用户] 阈值 6.5cm→1cm：贴面（d=2cm）不算成功，
        #   逼策略精确悬停（TCP 距质心 1cm 内=掌心悬停顶面上方 2cm）；饱和型避免二值跳变
        # [用户] z_std 0.02：TCP 只允许在 cube 质心及以上（质心以上满分，下方 2cm 内线性衰减）
        #   ——禁从下方接近/越过质心；贴面（TCP 质心下方 3cm）→ 0
        # [2026-09-12 平稳化] ① stable 软化：v_lin/v_ang 越过阈值后在 0.05 过渡带内线性衰减至 0
        #   （治"阈值边缘 0/1 flip → success 跳 4.0 → value loss 冲击"，当前 0.13 超警戒线）；
        #   ② d_std 0.01→0.02：close 过渡带放宽一倍，对距离噪声敏感度减半。
        params={"palm_dist_threshold": 0.015, "lin_vel_threshold": 0.20, "ang_vel_threshold": 0.20,
                "lin_vel_std": 0.05, "ang_vel_std": 0.05, "d_std": 0.02, "z_std": 0.02},
    )

@configclass
class CurriculumCfg:
    """课程学习（参考 SoftHand）：单次训练逐步激活抓取奖励，全程无 resume。

    ⚠️ [2026-09-14] 已停用（env_cfg.curriculum=None），且下面引用的 term 名均为旧版本
    （fingertip / tcp_gated_hand / lifted / contact_force 已不存在）——启用前必须先更新 term 名。

    阶段 1（0 ~ 30000 步 ≈ 100 episodes）: 手指奖励=0，策略先学接近（等同 Stage 1）
    阶段 2（30000 步后）: 激活手指弯曲/指尖接近/握紧奖励，学抓取
    阶段 3（60000 步后）: 激活指尖接触/抬起奖励，巩固抓握
    """

    # ===== 阶段 2：手指抓取（30000 步后开启）=====
    enable_finger_close = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "finger_close", "weight": 2.0, "num_steps": 30000},
    )
    enable_fingertip = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "fingertip", "weight": 2.0, "num_steps": 30000},
    )
    enable_tcp_gated_hand = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "tcp_gated_hand", "weight": 2.0, "num_steps": 30000},
    )

    # ===== 阶段 3：接触与抬起（60000 步后开启）=====
    enable_contact_force = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_force", "weight": 1.0, "num_steps": 60000},
    )
    enable_lifted = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "lifted", "weight": 2.0, "num_steps": 60000},
    )


@configclass
class CommandsCfg:
    """命令配置类。"""
    drill_pose = CmdTerm(
        class_type=mdp.DrillPoseCommand,
        resampling_time_range=(1000.0, 1000.0),
        debug_vis=True,
    )


@configclass
class TerminationsCfg:
    """终止条件配置：仅超时 + cube 掉落；成功不终止（逐步奖励）。

    [2026-09-14 清理] success 系列 / object_out_of_bounds / object_away_from_robot /
    debug_explosion 的注释块已删除（对应函数已清理，需用时从 git 历史恢复）。
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    drill_drop = DoneTerm(func=mdp.drill_dropped)


##
# Environment configuration
##


@configclass
class Ur5eDrillgraspEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Ur5eDrillgraspSceneCfg = Ur5eDrillgraspSceneCfg(num_envs=2048, env_spacing=4.0)
    # Simulation settings（摩擦系数 + PhysX GPU 接触上限）
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.5,  # 0.5→1.0：恢复旧系统值。0.5 时 Cube 易滑 → "稳住"难 → success 稀疏崩
            dynamic_friction=1.5,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2 ** 20,  # 256MB 物理接触
            gpu_max_rigid_patch_count=2 ** 23,  # 256MB 碰撞面片
            gpu_collision_stack_size=2 ** 29,  # 512MB 碰撞栈
        ),
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()
    curriculum = None  # 【观测模式】课程学习关闭：抓取/抬升奖励保持 weight=0，纯 Stage 1（reach）


    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        # 60Hz 实验证明接触太粗 → 稳定保持难 → success 封顶 ~10%（见 60Hz 运行日志）
        # 回到 120Hz 保证物理保真度（偏移已修复，这是学习可靠性的关键）
        self.decimation = 4
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation