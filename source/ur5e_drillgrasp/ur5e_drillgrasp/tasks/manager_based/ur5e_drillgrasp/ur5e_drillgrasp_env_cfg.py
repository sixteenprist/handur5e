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
            size=(0.07, 0.07, 0.07),  # v20: 6cm→7cm——更大目标，迫使手指张开包裹（希望自然带动中段关节3）
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,  # 保持 16（抓取稳定性需要，不动）
                solver_velocity_iteration_count=0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.5, 0.8)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.18, 0.11, 0.785),  # 桌面顶 0.75 + 半高 0.035，贴合桌面；y 0.10→0.11
        ),
    )

    # 【后续换电钻时取消下面注释，注释掉上面】
    # object = DRILL_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")

    # robot
    robot = DRILL_UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # --- 指尖接触传感器（已禁用：新系统上恒为 0 + 每步查询开销大，提速 20~50%）---
    # 修好传感器 / 需要触觉奖励时，取消下面注释并恢复 drill_ur5e.py 的 activate_contact_sensors=True
    # contact_thumb = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/hand/thumb4",
    #     update_period=0.0,
    #     history_length=0,
    #     debug_vis=False,
    #     track_pose=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    # )
    # contact_index = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/hand/index4",
    #     update_period=0.0,
    #     history_length=0,
    #     debug_vis=False,
    #     track_pose=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    # )
    # contact_middle = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/hand/middle4",
    #     update_period=0.0,
    #     history_length=0,
    #     debug_vis=False,
    #     track_pose=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    # )
    # contact_ring = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/hand/ring4",
    #     update_period=0.0,
    #     history_length=0,
    #     debug_vis=False,
    #     track_pose=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    # )
    # contact_little = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/hand/little4",
    #     update_period=0.0,
    #     history_length=0,
    #     debug_vis=False,
    #     track_pose=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    # )

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
    # position_scale=0.005 (10× 细化)，orientation_scale=0.005
    arm_action = mdp.OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ],
        body_name="base_link_1",
        body_offset=mdp.OperationalSpaceControllerActionCfg.OffsetCfg(
            pos=(0.03, -0.02, 0.06),  # TCP offset (B)；rewards/obs/reset 的 _BODY_OFFSET 已同步
            rot=(1.0, 0.0, 0.0, 0.0),  # base_link_1 局部坐标系 (B)，恒等四元数
        ),
        position_scale=0.01,  # 保持与 Stage 1 一致（0.005 会破坏续训策略的动作映射）
        orientation_scale=0.005,  # ±0.005rad 精细姿态
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            gravity_compensation=True,
            inertial_dynamics_decoupling=True,  # 惯性解耦，提升 OSC 稳定性
            motion_stiffness_task=(120.0, 120.0, 120.0, 80.0, 80.0, 80.0),  # 介于原(80/50)与(187/194)之间
            motion_damping_ratio_task=(0.7, 0.7, 0.7, 0.7, 0.7, 0.7),  # 参考  0.56，稍偏保守
        ),
    )

    # ===== 手指动作：力矩控制，20→12维耦合 =====
    # scale=0 会导致死通道 std/熵无限膨胀 → 熵奖励污染手臂 → success 下滑（已实测）
    # 改回活跃通道 + hand_vel 静止惩罚：手指仍静止，但通道有梯度约束，不污染
    hand_action = mdp.GroupedHandActionCfg(
        asset_name="robot",
        scale=0.05,
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

        # ---------------- 指尖触觉（接触力范数）----------------
        # 传感器已禁用（提速）；修好后再恢复
        # fingertip_force = ObsTerm(func=mdp.fingertip_contact_force, )

        # ---------------- 目标命令（后续阶段用）----------------
        goal_pose = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "drill_pose"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


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

    cache_object_init_pos = EventTerm(  # 记录物体初始位置，供 object_out_of_bounds 使用
        func=mdp.cache_object_init_pos_on_reset,
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

    # ===== 【Stage 1：桌面接近】Cube 留在桌面，手臂学习靠近 =====
    # 后续 Stage 2/3 时取消注释下面这行
    # reset_cube_to_palm = EventTerm(
    #     func=mdp.reset_cube_to_palm,
    #     mode="reset",
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )


@configclass
class RewardsCfg:
    """Reward 权重配置。

    三阶段课程学习：
      Stage 1（接近）: 只有 reach + success_stage1，手指不动
      Stage 2（抓取）: 激活 finger_close + fingertip + contact_force + tcp_gated_hand
      Stage 3（举升）: 再激活 lift + palm_height，目标举到特定高度
    """

    # ===== 通用 =====
    reach: RewTerm = RewTerm(
        func=mdp.reach_reward,
        weight=3.0,  # Stage 1 验证过的权重（能跑到 ~13% 成功）
        # v14: std 0.4→1.0。旧值在 ~1m 处 tanh 饱和（奖励 0.01/步、梯度 0.16/m），
        # 远场拉不动 → 策略卡在 1m、noise std 膨胀到 0.52 不学习（v13@268 success=0）。
        # std=1.0：1m 处奖励 0.71/步、梯度 1.25/m（8×），0.5m→1.6，0.3m→2.1，0.1m→2.7（×3 权重）
        params={"std": 1.0},
    )
    action_rate: RewTerm = RewTerm(
        func=mdp.action_rate_penalty,
        weight=-0.02,  # 参考  -0.02，促更平滑动作
    )
    joint_vel: RewTerm = RewTerm(  # 关节速度 L2 惩罚
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    tcp_velocity: RewTerm = RewTerm(
        func=mdp.tcp_velocity_penalty,
        # v15: 接近减速（原 0.0 关闭）——距 cube<15cm 后平滑减速到 ~6cm/s，
        # 把"直接撞上"变成柔和接触，减少 cube 位移（也让 success 的"稳定"更易满足）。
        # 权重不能太重：否则最后 10cm 走太慢，299 步 episode 内到不了成功区。
        weight=-0.002,
        params={"vel_std": 0.1, "gate_dist": 0.15},
    )
    tcp_orientation: RewTerm = RewTerm(
        func=mdp.tcp_orientation_penalty,
        weight=-0.05,  # -0.1→-0.05：再放松姿态约束，让策略敢动
        params={"ang_std": 0.15},
    )
    tcp_close_bonus: RewTerm = RewTerm(
        func=mdp.tcp_close_bonus_once,
        weight=1.0,  # 一次性奖励：TCP 首次贴近 Cube
        # v21: 4.5cm→4cm——v20 偏宽；4cm = 距 7cm cube 表面 0.5cm，真正轻触
        params={"distance_threshold": 0.040, "bonus_value": 5.0},
    )
    object_distance_penalty: RewTerm = RewTerm(
        func=mdp.object_distance_penalty,
        weight=0.0,  # 关闭：数值发散源（曾出现 88km 距离），且非抓取核心
        params={"threshold": 0.25},
    )

    # ===== 手指抓取（Stage 2：从 Stage 1 最优 checkpoint 续训，直接激活）=====
    # 手指奖励全部带 TCP 距离门控：手臂没贴近时给 0，不会破坏已学好的接近行为
    finger_close: RewTerm = RewTerm(
        func=mdp.finger_close_reward,
        weight=3.0,  # v16: 2→3，压过 success 的稳定约束，让"包住"比"保稳"更值
        # v19: max_flex 1.0→0.5——防无限加压把 cube 挤飞（v18@3053 崩溃根因），弯到够用即饱和
        # v21: 门控 10cm→8.5cm——v20 太宽；8.5cm = 距 7cm cube 表面 5cm，贴近才奖励弯曲
        params={"dist_threshold": 0.085, "max_flex": 0.5},
    )
    middle_flex: RewTerm = RewTerm(
        func=mdp.middle_flex_reward,
        weight=2.0,  # v24: 直接奖励中段关节3弯曲（index3/middle3/ring3/little3）——专治"中段不弯"
        params={"dist_threshold": 0.10, "max_flex": 0.5},
    )
    fingertip: RewTerm = RewTerm(
        func=mdp.fingertip_contact_reward,
        weight=3.0,  # v16: 2→3
        # v23: std 0.08→0.04——0.08 时指尖差 1-2cm 就拿到 82%，无"最后 1cm"梯度 → 中段不弯不碰；
        # 0.04 只奖励真正贴到（2cm→0.54、碰到→1.0），恢复接触梯度
        params={"std": 0.04, "gate_std": 0.15},
    )

    tcp_gated_hand: RewTerm = RewTerm(
        func=mdp.tcp_gated_hand_action_reward,
        weight=0.0,  # 与 finger_close 冗余（同为"贴近后奖励弯曲"），暂不启用
        params={"distance_std": 0.1, "flex_std": 0.3},
    )

    # ===== 指尖触觉（初始 weight=0，由课程学习在 60000 步后激活）=====
    contact_force: RewTerm = RewTerm(
        func=mdp.contact_force_reward,
        weight=0.0,  # 课程学习 → 1.0
        params={"force_threshold": 0.02, "deadzone": 0.005, "bonus_per_contact": 1.0},
    )
    excess_force: RewTerm = RewTerm(
        func=mdp.excess_force_penalty,
        weight=0.0,  # 不启用
        params={"threshold": 5.0},
    )

    # ===== 物体抬升（初始 weight=0，由课程学习在 60000 步后激活）=====
    lifted: RewTerm = RewTerm(
        func=mdp.object_lifted,
        weight=0.0,  # 课程学习 → 2.0
        params={"minimal_height": 0.825},
    )

    # ===== 举升（Stage 3 启用，当前 weight=0）=====
    lift: RewTerm = RewTerm(
        func=mdp.lift_reward,
        weight=0.0,  # Stage 3 → 2.0
        params={"gate_dist": 0.3, "gate_steep": 10.0, "flex_thresh": 0.3, "flex_steep": 5.0, "z_thresh": 1.25},
    )
    palm_height: RewTerm = RewTerm(
        func=mdp.palm_height_reward,
        weight=0.0,  # Stage 3 → 1.0
        params={"z_thresh": 1.30},
    )

    # ===== 成功（观测模式：纯 Stage 1，只要求手掌贴近 + 稳定，不要求手指/抬起）=====
    # 逐步发放、不终止；权重 30→3：保留每步"拉近+保持"梯度，双峰回报方差降 10×（9000→900）
    # 防 entropy 暴涨崩溃（v5/v9/v10 权重 30 时熵 16+ 即崩盘前兆）
    success_reward: RewTerm = RewTerm(
        func=mdp.success_stage1_reward,
        weight=3.0,
        # v21: 7.5cm→6.5cm——v20 太宽；6.5cm 表示真正包住（距 7cm cube 表面 3cm）
        # v23: 稳定阈值放宽 lin 0.05→0.10、ang 0.10→0.20——抓握必然推动 cube，
        # 严的"稳定"在惩罚"碰 cube"（碰就掉 3/步），是"手指不接触"的结构性根源
        params={"palm_dist_threshold": 0.065, "lin_vel_threshold": 0.10, "ang_vel_threshold": 0.20},
    )

@configclass
class CurriculumCfg:
    """课程学习（参考 SoftHand）：单次训练逐步激活抓取奖励，全程无 resume。

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
    """终止条件配置。

    Stage 1: success → success_stage1（手掌贴近 + 稳定，不要求手指）
    Stage 2: success → success（+ 手指闭合）
    Stage 3: success → success_stage3（+ 举升高度）
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    drill_drop = DoneTerm(func=mdp.drill_dropped)
    object_out_of_bounds = DoneTerm(       # 物体被推飞即终止
        func=mdp.object_out_of_bounds,
        params={"threshold": 1.0, "use_xy_only": True},
    )
    object_away_from_robot = DoneTerm(     # 物体远离机器人即终止
        func=mdp.object_away_from_robot,
        params={"threshold": 2.0},
    )
    # 成功不终止（旧系统验证：逐步奖励 + 跑满 299 步 + reach 成功；崩溃根因是 TCP 偏移错位）
    # success = DoneTerm(func=mdp.success_stage1)


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
            static_friction=1.0,  # 0.5→1.0：恢复旧系统值。0.5 时 Cube 易滑 → "稳住"难 → success 稀疏崩
            dynamic_friction=1.0,
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