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
    # v67: replicate_physics 保持 True（用户判断：drill_drop 不是穿透，是接触速度太快撞飞 cube，
    # 走 tcp_velocity 减速路线解决；replicate_physics=False 会降低 steps/s，不采用）
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
            size=(0.06, 0.06, 0.06),  # v71: 7cm→6cm（用户要求）；6cm 更小、更接近实际电钻握持尺寸
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
            pos=(-0.16, 0.175, 0.78),  # v71: z 0.785→0.78（6cm 正确落点 = 桌面 0.75 + 半高 0.03，治悬浮）
        ),
    )

    # 【后续换电钻时取消下面注释，注释掉上面】
    # object = DRILL_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")

    # robot
    robot = DRILL_UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # --- 指尖接触传感器（v48 恢复：用户要修传感器，让 env 知道抓没抓紧）---
    # 之前禁用因为新系统上恒为 0 + 每步查询开销大；配合 drill_ur5e.py 的 activate_contact_sensors=True
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

    # ===== [调试] TCP 位置 marker（红色小球跟随 TCP，viewer 里看 TCP 在哪）=====
    # v83b: 纯视觉无碰撞；由 EventCfg.update_tcp_marker(interval) 每步更新。
    #   v83d: radius 0.012→0.03——1.2cm 太小被手/手指遮挡看不见。
    # v84: 训练时注释（省每步写入开销）；play/debug 验证时解开此 asset + EventCfg.update_tcp_marker。
    # tcp_marker: AssetBaseCfg = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/TCPMarker",
    #     spawn=sim_utils.SphereCfg(
    #         radius=0.03,
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
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
        body_name="wrist_3_link",
        body_offset=mdp.OperationalSpaceControllerActionCfg.OffsetCfg(
            pos=(0.0, 0.07, 0.08),  # v92: (0,0.10,0.08)→(0,0.07,0.08)——用户实测 TCP 在掌心下 7-8cm 太深（"掌心距顶面 3cm"时 TCP 深入 cube 内、在质心下方被 tcp_too_deep 误罚）；y 减 3cm 后 TCP 在掌心下 ~5cm，目标态 TCP 在质心上方 1cm。rewards/obs/reset 已同步
            rot=(1.0, 0.0, 0.0, 0.0),  # wrist_3_link 局部坐标系 (B)，恒等四元数
        ),
        position_scale=0.01,  # 保持与 Stage 1 一致（0.005 会破坏续训策略的动作映射）
        orientation_scale=0.005,  # ±0.005rad 精细姿态
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            gravity_compensation=True,
            inertial_dynamics_decoupling=True,  # 惯性解耦，提升 OSC 稳定性
            motion_stiffness_task=(120.0, 120.0, 120.0, 80.0, 80.0, 80.0),  # 介于原(80/50)与(187/194)之间
            # v93: 阻尼 0.7→1.2——零动作漂移（debug 实测 wrist 零动作绕 z 转 ~22°）。
            #   pose_rel 无绝对锚点（目标=当前+0 增量），欠阻尼 0.7 放大漂移；
            #   1.2 过阻尼显著抑制，接近动作仍够快（299 步内到目标）。
            motion_damping_ratio_task=(1.2, 1.2, 1.2, 1.2, 1.2, 1.2),
        ),
    )

    # ===== 手指动作：力矩控制，20→12维耦合 =====
    # scale=0 会导致死通道 std/熵无限膨胀 → 熵奖励污染手臂 → success 下滑（已实测）
    # 改回活跃通道 + hand_vel 静止惩罚：手指仍静止，但通道有梯度约束，不污染
    # hand_action = mdp.GroupedHandActionCfg(
    #     asset_name="robot",
    #     scale=0.05,
    # )


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
    # update_tcp_marker = EventTerm(  # [调试] 每步把红球移到 TCP 位置；训练时注释
    #     func=mdp.update_tcp_marker,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),  # 每步触发（interval 事件要求显式指定，0 = 每步）
    #     params={"interval": 1},
    # )

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
      Stage 2（抓取）: 激活 finger_close + fingertip + opposition（对向夹持）
      Stage 3（举升）: 再激活 lift，目标举到特定高度
    """

    # ===== 通用 =====
    reach: RewTerm = RewTerm(
        func=mdp.reach_reward,
        # v68: weight 1.5→3.0、std 0.40→0.55——恢复 v63 验证过的成功组合（success 2.61）。
        #   v66 降 weight/std 是"治标"：当时误以为 reach 高导致 action_rate 炸；实际根因是
        #   手部无 clip/无限幅，已由 v66 actions.py（clip_range + max_delta）治本解决。
        #   现 TCP/USD/cube 都回到 v40 时代，reach 也应回到 v40/v63 的成功值。
        weight=3.0,
        # v14: std 0.4→1.0（远场拉动力）。
        # v43: std 1.0→0.6——目标附近梯度变陡：差 2cm 也付出代价 → 逼手下沉补偿 TCP 偏移
        # （v40 的 TCP 偏移 z 0.08 因 6.5cm 阈值+平梯度而没生效，手不贴面）
        # v45: std 0.6→0.55——再陡一点，配合 success 4.5cm 阈值让手更贴 cube
        # v66: std 0.55→0.40（v68 改回 0.55，见上）
        # v71: std 0.60→0.55（用户：8cm 太远、要 6cm）——success 阈值已回 6cm
        # v75: std 0.55→0.45——破解 success 徘徊（289~353 轮一直在 1.0~1.5 不涨）：
        #   std 0.55 时 6~15cm 梯度太平，策略停在 10~15cm"够用"没动力推进到 6cm。
        #   std 0.45 近端更陡：6cm→2.58、10cm→2.33（差 0.25/步），逼策略推进到 6cm 边界。
        #   配合 v74 防撞惩罚（palm_press/tcp_too_close），推进时撞不了。
        # v82: 新增 sat_dist——d<sat 时 reach 给满分 1.0（饱和），消除"越近越好"。
        # v91: sat 6→2cm + sat_transition 平滑（TCP 掌心下 7-8cm 时代的几何）。
        # v92: std 0.45→0.30、sat 2→1cm——TCP 改到掌心下 5cm 后，目标态"掌心距顶面 3cm"
        #   对应 TCP 距质心 1cm。std 0.45 在 1~10cm reach 全 0.9+（近端无梯度）；
        #   0.30 让 2~20cm 有清晰梯度（0.93→0.42）。1cm 饱和 + 1~2cm 过渡：
        #   掌心距顶面 3cm 达饱和，继续压 reward 掉 → 停在 cube 上方。
        params={"std": 0.55, "sat_dist": 0.01, "sat_transition": 0.01},
    )
    action_rate: RewTerm = RewTerm(
        func=mdp.action_rate_penalty,
        weight=-0.02,  # 参考  -0.02，促更平滑动作
    )
    # hand_action: RewTerm = RewTerm(
    #     func=mdp.hand_action_penalty,
    #     # v77: 手指伸直惩罚——play(model_400)：hand 碰 cube 时拇指指根被顶弯向掌心、
    #     #   回升后保持弯（旧版只罚动作=保持姿态，被碰弯后无"回伸直"梯度）。
    #     #   改为罚 max|finger joint_pos|（离伸直位 0 越远越罚），被碰弯后自动回伸直。
    #     #   weight -0.5：最弯关节 0.5rad 罚 0.25/步，伸直=0。
    #     #   ⚠️ Stage 2 激活手指奖励时必须 weight→0（否则与抓取弯曲冲突）。
    #     weight=-0.5,
    # )
    # (v80 误诊已撤回：用户 2026-08-27 确认碰 cube 的是手掌不是手指，根因是 rewards.py
    #  的 _BODY_OFFSET/_CUBE_HALF_SIZE 未随 v71 同步，v81 已改单一来源导入修复)
    joint_vel: RewTerm = RewTerm(  # 关节速度 L2 惩罚
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    tcp_velocity: RewTerm = RewTerm(
        func=mdp.tcp_velocity_penalty,
        # v73: 加强减速防撞——用户确认：手掌倾斜是"撞到 cube 后 hand 继续碰撞被碰歪"
        #   （tcp_orientation 惩罚大是症状，根因还是碰 cube，不动 tcp_orientation）。
        #   gate 0.15→0.10：只在最后 10cm 重减速（避开 v70 gate 0.20 过早减速拖慢接近的坑）；
        #   weight -0.002→-0.005、vel_std 0.05→0.06：接近最后阶段速度压得更狠。
        #   v15 的 -0.002/gate 0.15 在旧 TCP 几何下够用，现 TCP 偏移小（TCP 近=手近）需加强。
        #   v78: 轻微减速——play(model_400)：接近时"先碰 cube 一下再回升"（手指超调接触）。
        #   注意：不能太狠，否则最后接近太慢，299 步内到不了成功区。
        #   gate 0.10→0.12（更早进减速区）、weight -0.005→-0.006（轻加强），减轻接近动量。
        # v90: weight -0.006→-0.03、vel_std 0.06→0.05——297~300 轮新退化：episode length
        #   299→224、drill_drop 0→0.001（cube 被撞掉落）。旧 weight 下 0.5m/s 接近只罚 0.21/步
        #   （vs reach 3.0 微不足道），策略高速冲向 cube 撞飞。加强后：0.3m/s→0.9/步、0.5m/s→2.5/步，
        #   策略被迫在 12cm 内减速到 <0.3m/s。gate 保持 0.12 不提前（避 v70 拖慢接近坑）。
        weight=-0.03,
        params={"vel_std": 0.05, "gate_dist": 0.12},
    )
    palm_press: RewTerm = RewTerm(
        func=mdp.palm_press_penalty,
        # v74/v82/v86: 原防撞主力（weight -3.0）——罚 base_link_1 低于 cube 顶面。
        # v92: 关闭（weight 0）——debug 实测 base_link_1 在掌心下方 ~4-6cm，目标态
        #   "掌心悬停 3cm"时 base_link_1 已低于顶面（误判压入、罚 ~1.3/步），会阻止
        #   用户要的悬停姿态。防压入已由 tcp_too_close（z 深度版）完整接管：
        #   TCP 低于质心（掌心压到 2cm 以下）才罚，目标态（TCP 质心上方 1cm）不罚。
        weight=0.0,
        params={"clearance": 0.0, "pen_std": 0.015, "xy_gate": 0.06},
    )
    tcp_too_close: RewTerm = RewTerm(
        func=mdp.tcp_too_close_penalty,
        # v74/v86: 欧氏距离版（min_dist 5cm）——目标态反被罚 + 穿质心 U 形诱导压 cube（v91 根因）。
        # v91: 改单调 z 深度版——只罚 TCP 低于质心（depth>0）且水平在 cube 内（无 U 形）。
        # v92: weight -1.5→-3.0——TCP 掌心下 5cm 后目标态（掌心 3cm）TCP 在质心上方 1cm
        #   （depth=-1cm）不罚；但掌心 1~2cm 时 reach/success 仍饱和（欧氏 d=|h-2| 太小），
        #   只有 z 深度惩罚能区分——加强后：掌心 2cm→0、1cm→1.5/步、贴顶面→3/步、压入 1cm→4.5/步。
        weight=-3.0,
        params={"min_dist": 0.0, "pen_std": 0.02, "xy_gate": 0.06},
    )
    tcp_orientation: RewTerm = RewTerm(
        func=mdp.tcp_orientation_penalty,
        # v71: weight -0.01→-0.02、ang_std 0.3→0.2——用户确认初始姿态就是"手掌朝下平放"，
        #   是目标姿态，不允许大幅偏离（77 轮早期平均偏离 21° 被判定过松）。
        #   加强后：偏离 11.5° 内基本自由（探索不受限），15-20°+ 有效压制（0.04-0.07/步），
        #   明显偏离会被拉回平放姿态。
        #   仍比 v62（-0.03/0.15）温和约 1/3，避免重蹈"乱扭→被罚→更乱"失控；
        #   若从零训练再现 v62 模式（entropy 涨 + 姿态乱），再降回 -0.01/0.3。
        # v82: weight -0.02→-0.05——治 play 观察"hand 绕 z 轴转"。本函数是平方惩罚
        #   (ang/ang_std)²（与 Softhand tcp_orientation_deviation_from_init 一致）：
        #   绕 z 转 20°(0.35rad) 旧权重 -0.02 罚 0.061/步，-0.05 后罚 0.153/步、
        #   30° 罚 0.34/步——平方型对持续大偏离惩罚陡增，累积 299 步很可观。
        weight=-0.05,
        params={"ang_std": 0.2},
    )
    tcp_close_bonus: RewTerm = RewTerm(
        func=mdp.tcp_close_bonus_once,
        weight=1.0,  # 一次性奖励：TCP 首次贴近 Cube
        # v21: 4.5cm→4cm——v20 偏宽；4cm = 距 7cm cube 表面 0.5cm，真正轻触
        # v71: 4cm→8cm——防贴脸（std 方案下近端梯度平，4cm 一次性 bonus 会引诱策略
        #   穿过 success 边界贴到 4cm）；8cm 让策略在接近途中先拿 bonus 再停在边界。
        # v75: bonus 5→10——里程碑激励。
        # v91: threshold 8cm→2cm——与 reach/success 饱和点一致。
        # v92: 2→1.5cm——新几何目标态 TCP 距质心 1cm，首次进入 1.5cm 悬停区给 bonus。
        params={"distance_threshold": 0.015, "bonus_value": 10.0},
    )

    # ===== 手指抓取（Stage 2：从 Stage 1 最优 checkpoint 续训，直接激活）=====
    # 手指奖励全部带 TCP 距离门控：手臂没贴近时给 0，不会破坏已学好的接近行为
    finger_close: RewTerm = RewTerm(
        func=mdp.finger_close_reward,
        weight=0.0,  # v59: 重跑 Stage 1（纯 reach）暂关；v16 原 3.0——先学手掌悬浮接近
        # v19: max_flex 1.0→0.5——防无限加压把 cube 挤飞（v18@3053 崩溃根因），弯到够用即饱和
        # v21: 门控 10cm→8.5cm——v20 太宽；8.5cm = 距 7cm cube 表面 5cm，贴近才奖励弯曲
        # v26b: max_flex 0.7→0.5 回退——v26 的 0.7 引起价值函数冲击，47 轮内全面回归(value loss 0.13)
        # v47: 0.5→0.7——test_lift 证明策略不主动握紧（target≈0，手指被顶弯）；
        # 弯曲奖励在 0.5 饱和 → 策略停在 0.5、action≈0 → 目标偏差≈0 → 握力=0。
        # 提高饱和点给"弯更深"的梯度，产生持续目标偏差=持续握力（相对位置控制内）
        params={"dist_threshold": 0.085, "max_flex": 0.7},
    )
    middle_flex: RewTerm = RewTerm(
        func=mdp.middle_flex_reward,
        weight=0.0,  # v59: 重跑 Stage 1 暂关；v24 原 2.0——先学手掌悬浮
        # v26b: max_flex 0.7→0.5 回退——与 finger_close 同步（见上）
        # v47: 0.5→0.7——与 finger_close 同步（让中段弯曲也有持续梯度）
        params={"dist_threshold": 0.10, "max_flex": 0.7},
    )
    # v44: 掌根(关节2)弯曲——专治 ring/little 靠指尖(j4)硬够的蜷缩爪
    # play(3800): ring/little 的 j2 弯不够、j4 过弯（指尖独弯去够表面）。
    # finger_close 是全指聚合(0.7min+0.3mean)，ring/little 的 j2 提升被 min 稀释；
    # 本项定向给 ring/little 掌根压力：掌根先弯→整指包住 cube→j4 自然落在侧面。
    # v45: 扩展到四指——play(3907) 四指 j2 都可再弯一点（整体包络更好）
    # v46: weight 1.5→1.8——play(4100) 关节2 还可再微弯（已 95% 饱和在望，微升压力）
    # v47: max_flex 0.5→0.7——与 finger_close 同步
    base_flex: RewTerm = RewTerm(
        func=mdp.base_flex_reward,
        weight=0.0,  # v59: 重跑 Stage 1 暂关；v46 原 1.8——先学手掌悬浮
        params={"dist_threshold": 0.10, "max_flex": 0.7, "fingers": ("index", "middle", "ring", "little")},
    )
    # v46: 拇指指尖定向贴近——play(4100) 四指已到位但拇指没碰到 cube。
    # fingertip 是全指聚合(0.7mean+0.3min)，thumb 梯度被 mean 稀释；本项单独给拇指压力
    thumb_contact: RewTerm = RewTerm(
        func=mdp.thumb_contact_reward,
        weight=0.0,  # v59: 重跑 Stage 1 暂关；v46 原 1.5——先学手掌悬浮
        params={"std": 0.04, "gate_std": 0.15},
    )
    fingertip: RewTerm = RewTerm(
        func=mdp.fingertip_contact_reward,
        weight=0.0,  # v59: 重跑 Stage 1 暂关；v16 原 3.0——fingertip 曾诱惑策略压深（指尖近=高分），先学悬浮
        # v23: std 0.08→0.04——0.08 时指尖差 1-2cm 就拿到 82%，无"最后 1cm"梯度 → 中段不弯不碰；
        # 0.04 只奖励真正贴到（2cm→0.54、碰到→1.0），恢复接触梯度
        # v46: std 0.04→0.035——play(4100) 指尖还差一点点，再收紧让最后 1cm 梯度更尖
        params={"std": 0.035, "gate_std": 0.15},
    )
    finger_ratio: RewTerm = RewTerm(
        func=mdp.finger_ratio_reward,
        weight=0.0,  # v59: 重跑 Stage 1 暂关；v38 原 1.5——先学手掌悬浮
        # v42: ratio_target 0.4→0.6——0.4 太狠卡死指尖接触（finger_ratio≈0、fingertip 0.30 上不去）；
        # 0.6 保留"指尖比中段弯得少"的自然顺序，又给指尖留出接触空间
        # v55: ratio_target 0.6→0.85——曾导致 test_sensors 四指指尖全弯死(1.57)蜷缩爪
        # v58: 0.85→0.6 收回——用户物理洞察：夹取靠指腹发力，j4 应 ≤0.6×j3 保持指腹平贴侧面。
        # v57 手掌悬浮顶面上方后中段会弯（包络），j4 有空间，不再卡死接触
        params={"ratio_target": 0.6, "dist_threshold": 0.12},
    )
    # v55: 手指过弯惩罚——max_flex 只是饱和点不是上限，策略把中段弯到 1.57 锁死手势
    # （test_sensors: ind(m1.57)），无法侧摆/调整形成对向。惩罚超过 thresh 的部分。
    # v81: func 从 mdp.overflex_penalty 修正为 mdp.fingertip_overcurl_penalty——
    #   函数在 v36 已改名（本地 rewards.py 只有 fingertip_overcurl_penalty），
    #   env_cfg 引用未同步，远程同步后首次暴露 AttributeError（weight=0 不参与训练，纯引用修复）。
    overflex: RewTerm = RewTerm(
        func=mdp.fingertip_overcurl_penalty,
        weight=0.0,  # v59: 重跑 Stage 1 暂关；v55 原 -0.5——先学手掌悬浮
        params={"thresh": 1.0, "dist_threshold": 0.12},
    )

    # ===== 指尖触觉（v48 激活：传感器已恢复，直接奖励"抓没抓紧"）=====
    # v47: grasp_contact 激活（无需传感器）——间接检测，先保留
    # v50: grip_force 替代 contact_force（连续力）——v49 诊断：接触力仅 0.03~0.09N，
    # 二值奖励(>0.02N 满分)无"压多紧"梯度，策略擦到就停。连续力 1-exp(-f/σ) 全程有梯度。
    grasp_contact: RewTerm = RewTerm(
        func=mdp.grasp_contact_reward,
        weight=0.0,  # v59: 重跑 Stage 1 暂关；v53 原 1.5——先学手掌悬浮
        # v53: bonus_per_step 2.0→0.0——v51@4580 与 v52@4815 两次崩溃的共同驱动者：
        # 隐藏强奖励（weight1.5×2.0≈3.0/步）驱动策略挤压 cube 制造位移，牺牲稳定（success 2.81→1.73）。
        # 只保留间接检测（flex+位移），不再给持续高分
        params={"flex_threshold": 0.4, "displacement_threshold": 0.003, "bonus_per_step": 0.0},
    )
    # v52: opposition 替代 grip_force（对向夹持 = 力封闭平滑形式）
    # v51 诊断：四指全压 +x 面 → 力全同向 → 无对向 → cube 侧滑。grip_force（总力）
    # 会被单侧压骗到高分。opposition = min(F_+x,F_-x)+min(F_+y,F_-y)：只有相对面
    # 都有力才给分，单侧压=0 分。这才引导形成真正能夹住的形式。
    opposition: RewTerm = RewTerm(
        func=mdp.opposition_reward,
        weight=0.0,  # v56: 暂时关闭——先从 model_3300 学正确手势（侧摆+适度弯+指尖参与）
        # v52-v54 三次崩溃教训：opposition 需要"正确手势"作为前提，在错误手势上
        # 只会引导策略乱动崩掉。v55 的手势修正（侧摆权重1.0 + overflex + 指尖参与）
        # 是关节权重的直接引导，不需要 opposition。手势验证通过后再决定是否开启。
        # v81b: 移除 lin_vel_std/ang_vel_std——v54 注释称保留"手端稳定约束"，但
        #   opposition_reward 最终未实现这两个参数（签名仅 gate_dist/scale），
        #   残留参数导致 RewardManager 初始化 ValueError（weight=0 也会解析参数）。
        params={"gate_dist": 0.12, "scale": 1.0},
    )
    # 已废弃：grip_force/contact_force/excess_force（v50-v52 演进中被 opposition 取代，函数保留在 rewards.py）

    # ===== 物体抬升（初始 weight=0，由课程学习在 60000 步后激活）=====
    lifted: RewTerm = RewTerm(
        func=mdp.object_lifted,
        weight=0.0,  # 课程学习 → 2.0
        params={"minimal_height": 0.825},
    )

    # ===== 举升（Stage 3）=====
    # v27: z_thresh 0.85→0.815（cube 只抬 3cm 就出梯度）——v26 崩盘的根因之一：
    # 6.5cm 阈值太稀疏，策略学不会"边保持抓握边抬臂"，盲试中丢抓握→0 信号→崩。
    # 3cm 贴近现有抓握能力，让"轻轻带起"就有连续奖励。
    # v28: z_thresh 0.815→0.82——cube 变大到 8cm 后静止中心 z=0.79，保持"离桌 3cm"语义（0.79+0.03）
    lift: RewTerm = RewTerm(
        func=mdp.lift_reward,
        # v31: 保持 0 关闭；z_thresh 回 0.815（7cm cube 静止 0.785 + 3cm）。等 test_lift 确认可举再激活
        weight=0.0,
        params={"gate_dist": 0.3, "gate_steep": 10.0, "flex_thresh": 0.3, "flex_steep": 5.0, "z_thresh": 0.815},
    )
    # palm_height 已删除（v26 崩溃元凶：奖励空手抬高、诱导丢抓握，不再使用）

    # ===== 成功（观测模式：纯 Stage 1，只要求手掌贴近 + 稳定，不要求手指/抬起）=====
    # 逐步发放、不终止；权重 30→3：保留每步"拉近+保持"梯度，双峰回报方差降 10×（9000→900）
    # 防 entropy 暴涨崩溃（v5/v9/v10 权重 30 时熵 16+ 即崩盘前兆）
    success_reward: RewTerm = RewTerm(
        func=mdp.success_stage1_reward,
        # v88: 3.0→2.0——282 轮日志 value loss 0.47（success 出现后价值震荡，v87 平滑仍压不住）。
        #   weight 线性缩放不改变最优行为，只减小价值函数波动；可续训不用重训。
        weight=2.0,
        # v21: 7.5cm→6.5cm；v23: 稳定阈值放宽
        # v43: palm_dist_threshold 6.5cm→5cm——TCP 必须贴到 5cm 内 → 配合 v40 的 TCP 上移，
        # 手必须再下沉 ~1.5cm 才能满足 → TCP 偏移真正改变 hand-cube 相对距离
        # v45: 5cm→4.5cm——play(3907) 希望 hand 再靠近 cube 一点（收紧阈值，逼更贴）
        # v57: 4.5cm→7cm——test_sensors 实测 4.5cm 阈值下策略把手掌压到 cube 中部
        # （palm_z 0.78≈质心），手指失去从上方包住侧面的空间 → 指尖悬空弯死(1.57)碰不到 cube。
        # TCP 在手掌上方 8cm，TCP 距质心 7cm ⇒ 手掌悬浮在 cube 顶面上方 ~3.25cm，
        # 给手指留出"从中段弯下包住侧面"的夹取空间。
        # v70: 去掉 v69 的方向约束——用户要求不强制悬浮，就是让 TCP 接近 cube（参考点必须 TCP，不改 palm）。
        #   v69: 4.5cm→7cm→6cm 收紧 + 方向约束（已去掉）
        # v71: 用户确认 8cm 太远、要 6cm——阈值 0.090→0.060（回到 6cm 边界），
        #   配合 reach std 0.60→0.55（近端梯度更陡）逼策略推进到 6cm。
        # v79: 6cm→5cm——play 用户觉得离 cube 还差一点（v80 已改回 6cm：5cm 太近）
        # v80: 5cm→6cm——用户 play 后觉得 5cm 太近、改回 6cm。
        # v87: success 二值→饱和型（rewards.py success_stage1_reward 加 d_std 过渡带）——
        #   消除 0/1 跳变对价值函数冲击。
        # v91: threshold 6cm→2cm、d_std 0.02→0.01——与 reach 饱和点一致。
        # v92: threshold 2→1cm——TCP 掌心下 5cm 后目标态（掌心距顶面 3cm）TCP 距质心 1cm：
        #   1cm 内满分；1~2cm 过渡提供"别再近"梯度（配合 tcp_too_close -3.0）。
        params={"palm_dist_threshold": 0.010, "lin_vel_threshold": 0.10, "ang_vel_threshold": 0.20, "d_std": 0.01},
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

    # ===== 阶段 3：接触与抬起（60000 步后开启）=====
    enable_opposition = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "opposition", "weight": 2.0, "num_steps": 60000},
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