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
            size=(0.06, 0.06, 0.06),
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
            pos=(-0.16, 0.175, 0.78),  # 桌面顶 0.75 + 半高 0.035，贴合桌面；y 0.10→0.11
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
    # position_scale=0.005 (10× 细化)，orientation_scale=0.005
    arm_action = mdp.OperationalSpaceControllerActionCfg(
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
        position_scale=0.01,  # 保持与 Stage 1 一致（0.005 会破坏续训策略的动作映射）
        orientation_scale=0.005,  # ±0.005rad 精细姿态
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            gravity_compensation=True,
            inertial_dynamics_decoupling=True,  # 惯性解耦，提升 OSC 稳定性
            motion_stiffness_task=(200.0, 200.0, 200.0, 194.0, 194.0, 194.0),
            # [2026-09-02] 位置 200 不动，姿态 180→194（参考 SoftHand UR5e）
            # [2026-09-02] 位置/姿态分开设阻尼：位置 1.5 治下沉，姿态 0.56 治侧翻（欠阻尼响应快，参考 SoftHand）
            motion_damping_ratio_task=(1.5, 1.5, 1.5, 0.56, 0.56, 0.56),
        ),
    )
    # ===== 手指动作 =====
    # [Stage 1 2026-09-02] scale=1.5 保持与 Stage 2 一致（消除动作映射切换导致的分布偏移）：
    #   旧方案 Stage 1 scale=0 锁手指 → Stage 2 scale=1.5 打开，动作映射+奖励双切换，
    #   导致 Stage 1 学到的手指权重是纯噪声（动作无物理效果），Stage 2 打开后整个策略退化（用户观察）。
    #   新方案 Stage 1 就用 scale=1.5（手指物理自由），靠 hand_action_mag(-0.2) 逼策略输出 0 保持伸直，
    #   手指权重学到"输出 0"这个有意义的基础；Stage 2 只需激活 finger 奖励 + 撤 hand_action_mag，
    #   scale 不变 → 单重偏移，平滑过渡。
    # [Stage 2 2026-09-01] bias 暂不用（保持 0）：thumb1 初始 -30°→-10° 已修复物理。
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

        # ---------------- 指尖触觉（接触力范数）----------------
        # [v71] 注释掉：该观测把维度从 109 顶到 114，导致旧 checkpoint 无法加载（维度不匹配）。
        #   指尖力仍由 finger_contact 奖励内部直接读 sensor（fingertip_contact_force）使用，
        #   不需要作为观测喂给策略（策略靠 fingertip_to_cube 贴面距离 + 奖励梯度已足够）。
        # fingertip_force = ObsTerm(func=mdp.fingertip_contact_force, )

        # ---------------- 目标命令（后续阶段用）----------------
        goal_pose = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "drill_pose"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 特权观测组（借鉴 SoftHand 的非对称 actor-critic）。

        映射关系见 agents/rsl_rl_ppo_cfg.py 的 obs_groups：
          actor ← policy 组（109 维，维度不变，旧 actor 权重可复用）
          critic ← policy + critic 组（109 + 14 = 123 维）

        只放"仿真里拿得到、真机/策略端拿不到或没必要拿"的真值信息，
        让 value 估计更准（success 依赖 cube 稳定性与接触力，而 actor 看不到它们）。
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

        # ---------------- 指尖接触力真值（5）----------------
        # [v71] 为保 109 维 checkpoint 兼容从 policy 组移除的 fingertip_force，
        # 移入 critic 组两全其美：finger_contact / grip / fingertip_press 奖励
        # 都建立在接触力上，critic 看到力 → 优势函数质量更高。
        fingertip_force = ObsTerm(func=mdp.fingertip_contact_force, )

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
        # [用户] std 0.45→1.0（回归 v14）：std 只改梯度随距离的分布，不移动悬停目标
        #   （悬停由 success 1cm 饱和型驱动）。当初收到 0.45 是为"贴面→悬停"额外梯度，
        #   现已被 success 1cm 阈值取代（贴面 d=2cm 拿 0 分）→ 放宽回 1.0 恢复远场梯度
        #   （1m 外不再饱和），加速接近收敛；近端 2cm→0 推力 0.13→0.06/步，由 success 覆盖
        # [用户] z_std 0.02：TCP 只允许在 cube 质心及以上（与 success 方向约束一致）
        params={"std": 1.0, "z_std": 0.02},
    )
    action_rate: RewTerm = RewTerm(
        func=mdp.action_rate_penalty,
        weight=-0.05,  # [2026-09-03] -0.02→-0.05：治抓取时手指颤动——只罚 a_t−a_{t-1} 变化量，
        #   不罚恒定弯曲（抓取允许稳定保持弯曲），只压"来回抖"的高频颤动。
    )
    hand_action: RewTerm = RewTerm(
        func=mdp.hand_action_penalty,
        # [v72] 从零训练熵爆 32.8 修复：-0.5 太重 + 门控 1cm 太严 → 手指 12 维在接近阶段零梯度 → 熵爆。
        # [Step 1 2026-09-01] -0.05→-0.3：加强伸直压力。
        # [Stage 1 锁手指 2026-09-01] -0.3→0.0：scale=0 后手指物理锁死（相对位置控制 target=joint_pos+0，
        #   保持初始伸直），手指不弯 → 本惩罚恒 0；即使被 cube 顶弯，动作归零也无法回伸直，梯度无用。
        #   伸直改由 scale=0 物理保证，熵漂移由 hand_action_mag 防。Stage 2 打开手指时再恢复本项。
        weight=0.0,
    )
    # [v96 2026-09-01] 手指动作幅度惩罚——逼手指输出 0（静止），给无目标的手指 12 维提供确定性约束。
    #   根治：entropy_coef 熵奖励下手指 12 维无梯度 → 熵漂移 → 熵爆（v71:32.8 / 旧训练:29 / 本次:34.3）。
    #   hand_action(-0.3) 只罚弯曲量（一阶矩），手指仍可抖动（高熵）；本项罚动作平方（二阶矩）逼静止。
    # [v99 2026-09-01] -0.5→-0.2：-0.5 太硬，手指 12 维 σ 被压进负熵区（σ<0.242），总 entropy 30~40 iter 即掉负
    #   且持续减小，熵自校正被压制，手臂也快速塌到次优解（reach 卡 1.8~1.9，与 entropy_coef 0.001 时代同款过早收敛）。
    #   回到 -0.2：虽会慢熵漂移（300+ iter 到 11），但 reach 能爬到 2.4+，配合"reach 2.4 即切 Stage 2"可用。
    #   （根本解：只对手臂 6 维算 entropy bonus，让手指熵不进入 loss，见 rsl_rl PPO 子类化方案）
    hand_action_mag: RewTerm = RewTerm(
        func=mdp.hand_action_magnitude_penalty,
        weight=-0.2,  # [2026-09-03] 0.0→-0.2（加 TCP 门控）：抓取奖励只在 TCP 3cm 内给梯度，
        #   远处手指 12 维无梯度 → 熵漂移 → 熵爆（entropy 10.3）。恢复幅度惩罚治熵爆；
        #   gate_dist=0.03 门控：远→罚（逼手指 0），近→放开（不干扰抓取弯曲）。
        params={"gate_dist": 0.05},
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
        # [加强] -0.002→-0.05、vel_std 0.1→0.05：0.3m/s→1.08/步、0.5m/s→3.0/步，
        #   压住"接近速度过快"（用户 1500 轮观察）
        weight=-0.05,
        params={"vel_std": 0.05, "gate_dist": 0.15},
    )
    tcp_orientation: RewTerm = RewTerm(
        func=mdp.tcp_orientation_penalty,
        weight=-0.05,  # -0.1→-0.05：再放松姿态约束，让策略敢动
        # [2026-09-02] ang_std 0.25→0.15：play 观察策略"抬手腕空弯"（手腕偏离 38° 拿 finger_close 分，
        #   手指离 cube 侧面变远夹不到）。0.25 太松，38° 只罚 -0.35/步，挡不住抬手腕收益。
        #   0.15（8.6°）：38° 偏离罚升到 -0.97/步，让"抬手腕换弯曲分"变成亏本买卖。
        params={"ang_std": 0.25},
    )
    tcp_close_bonus: RewTerm = RewTerm(
        func=mdp.tcp_close_bonus_once,
        weight=1.0,  # 一次性奖励：TCP 首次贴近 Cube
        # v21: 4.5cm→4cm——v20 偏宽；4cm = 距 7cm cube 表面 0.5cm，真正轻触
        # [用户] bonus 8.0→5.0（撤回）：8.0 一次性尖峰在策略大量穿越 4cm 边界时密集发放，
        #   → 稀疏大奖励 → value loss 爆 28.8（952→1052 iter）→ 策略退化。回 5.0 控尖峰。
        params={"distance_threshold": 0.040, "bonus_value": 2.0},
    )
    # object_distance_penalty: RewTerm = RewTerm(
    #     func=mdp.object_distance_penalty,
    #     weight=0.0,  # 关闭：数值发散源（曾出现 88km 距离），且非抓取核心
    #     params={"threshold": 0.25},
    # )

    # ===== 手指抓取（Stage 2 简化版 v71：结果导向，4 项）=====
    # 正确弯曲 = 指尖贴面 × 指尖有力（finger_contact）；正确抓取 = 力封闭（grip）。
    # 密集引导：touch 是距离连续梯度（贴面有 base=0.4 底分），force/grip 做稀疏加分。
    # [Stage 2 2026-09-01] finger_close 关节角弯曲奖励：给"弯曲量"直接密集梯度，
    #   打破"伸直→不贴面→无梯度→伸直"的 bootstrap 死循环（finger_contact 只在贴面给分太稀疏）。
    # [2026-09-02] 硬门控→软门控 + 门控 8cm→3cm（gate_std=0.03）+ weight 1.5→0.5：
    #   finger_contact 已 0.49 起来，finger_close 退居二线，减弱过程导向副作用
    #   （抬手腕空弯/关节3过弯/抓取时颤动），真悬停才奖励弯曲。
    finger_close: RewTerm = RewTerm(
        func=mdp.finger_close_reward,
        weight=0.5,  # [2026-09-05] 1.0→0.5：精简过程奖励——抓取已成型，弯曲引导退居二线，减少对 finger_contact/grip 的拉扯
        # [2026-09-04] gate_std 0.03→0.08：弯曲应早于接触（时序：接近15cm→弯曲8cm→接触5cm→力封闭1.5cm）；
        #   max_flex 0.5→0.6：给"贴面需要更多弯曲"留余量（有层级门控+纯力判据约束，不会过度蜷缩）。
        params={"gate_std": 0.08, "max_flex": 0.6},
    )
    finger_reaching: RewTerm = RewTerm(
        func=mdp.finger_reaching_reward,
        weight=1.0,  # [2026-09-05] 2.0→1.0：精简过程奖励——接近引导减半，bootstrap 使命已完成，减少拉扯
        params={"touch_std": 0.03, "dist_threshold": 0.15, "d_std": 0.05},
    )
    finger_contact: RewTerm = RewTerm(
        func=mdp.finger_contact_reward,
        weight=3.0,  # [2026-09-04] 纯力判据（hysteresis 滞回）：sensor 有力才算接触，杜绝"接近不接触"虚高
        # [2026-09-04] dist_threshold 0.15→0.05：接触确认应晚于弯曲（时序：接近15→弯曲8→接触5→力封闭1.5cm）
        params={"deadzone": 0.005, "force_on": 0.04, "force_off": 0.02, "dist_threshold": 0.05, "d_std": 0.02},
    )
    # # [2026-09-03] 定向小指/无名指掌根弯曲：切抓握后食中指已弯贴面，小指无名指仍不弯，
    # #   finger_contact 的 min 短板（0.3）梯度不足 + hand_action_mag 压弯曲成本 → 短手指弯曲收益<成本。
    # #   本项直接给 ring/little 掌根关节2 弯曲梯度（0.5*mean+0.5*min），绕开贴面距离门槛，
    # #   先让短手指弯下来；贴面有力仍由 finger_contact 接力。
    # ring_little_flex: RewTerm = RewTerm(
    #     func=mdp.base_flex_reward,
    #     weight=1.0,
    #     params={"dist_threshold": 0.10, "max_flex": 0.5, "fingers": ("ring", "little")},
    # )
    # 正确抓取 = 力封闭（对向两面力取 min）。[Stage 2 2026-09-02] 0.0→1.0：激活力封闭奖励。
    grip: RewTerm = RewTerm(
        func=mdp.opposition_reward,
        weight=3.0,  # [2026-09-03 抓握阶段] 0.0→3.0：绕位完成，恢复力封闭奖励。
        # [2026-09-04] scale 0.5→0.1：实测指尖接触力仅 0.03~0.09N，0.5 时 tanh 几乎不响应；
        #   0.1 让 0.1N 对向力就有 tanh(1)=0.76 分，匹配接触力量级。力方向已改 cube 局部 y（见 rewards）。
        # [2026-09-05] scale 0.1→0.05：逼更大对向力（0.08N→tanh(1.6)=0.92，0.2~0.3N 才近满分）；
        #   配合 rewards.opposition_reward 里的 EMA 平滑滤噪，防 scale 变小时噪声被放大。
        # [2026-09-06] scale 0.05→0.08：治后期熵漂移（8500 崩）——0.05 时 opp≈0.06N 落在 tanh(1.2)=0.83
        #   半饱和段（梯度 sech²≈0.28 弱）→ 奖励平台化 → 优势趋零 → 熵奖励主导 → σ 漂移崩。
        #   0.08 让同样 opp 落到陡坡段 tanh(0.75)=0.64（梯度翻倍），满分需 opp≈0.24N，
        #   抓取后仍有持续加压梯度；配合 stiffness 60（opp 能上 0.1N+），掉分可爬回。
        params={"gate_dist": 0.015, "scale": 0.05, "d_std": 0.01},
    )
    # [2026-09-03] 拇指对侧奖励：引导拇指绕到四指对侧（力封闭的几何前提）。
    #   grip 是"力的稀疏确认"（拇指绕到位前恒 0，无梯度），本项给"几何密集引导"
    #   （接触前就有梯度）——互补：本项把拇指引到位，grip 确认夹住。
    thumb_opposition: RewTerm = RewTerm(
        func=mdp.thumb_opposition_reward,
        weight=2.0,
        params={"gate_std": 0.05},
    )
    # [v69][SoftHand 借鉴] 指尖合力向下分量惩罚——防"手指把 cube 压向桌面"的下压作弊。
    #   世界系 Z：向下合力 >0.02N 才计罚；水平对向夹持（力封闭）Z 分量小 → 少罚。
    # [堵洞 2026-09-01] 小指/无名指弯下来压 cube 提高 success（隐性收益 +3.0/步 > hand_action -0.24/步），
    #   原 gate_dist 1cm 太严（悬停时 TCP 距 cube ~1.5cm，门控恒不满足 → 惩罚从未触发，日志恒 0）。
    #   放宽 gate_dist 1cm→4cm 覆盖整个悬停区；weight -0.1→-0.5 让"手指压 cube"变成亏本买卖。
    fingertip_press: RewTerm = RewTerm(
        func=mdp.fingertip_press_penalty,
        weight=-0.2,  # [2026-09-03] -0.5→-0.2：配合 tanh 软饱和，单步惩罚上限降到 -0.2（原 -5），
        #   进一步降低接触力瞬态尖峰对 value 的冲击（黄线 Step 1800~2200 崩塌触发源）。
        params={"force_std": 1.0, "deadzone": 0.02, "gate_dist": 0.04},
    )
    # [Stage 2 2026-09-01] 手掌最小距离惩罚：堵"压近手掌"hack。
    #   finger_contact 只按指尖→表面距离计分，策略学会压近手掌让指尖贴面（不弯曲手指），
    #   手掌从悬停(2cm)变贴压(<2cm) → thumb1 掌根段侵入 cube 上方、碰 cube 上表面。
    #   本项在手掌表面距离 < 2cm 时平方惩罚（2cm→0、1cm→-2、0cm→-8 每步），逼手掌保持悬停。
    # palm_proximity: RewTerm = RewTerm(
    #     func=mdp.palm_proximity_penalty,
    #     weight=-2.0,
    #     params={"surface_threshold": 0.02, "std": 0.01},
    # )
    # [2026-09-05] 手掌压 cube 惩罚（启用）：堵"手掌撞 cube"hack。手掌 z 低于 cube 顶面(clearance 内)且
    #   水平投影在 cube 内(xy_gate)时惩罚——正常抓取手掌在 cube 上方悬停不触发，只有"手掌撞/压穿 cube"才罚。
    # palm_press: RewTerm = RewTerm(
    #     func=mdp.palm_press_penalty,
    #     weight=-2.0,
    #     params={"clearance": 0.005, "pen_std": 0.015, "xy_gate": 0.04},
    # )
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
        # [用户] 阈值 6.5cm→1cm + d_std 0.01 过渡带：贴面（d=2cm）不算成功，
        #   逼策略精确悬停（TCP 距质心 1cm 内=掌心悬停顶面上方 2cm）；饱和型避免二值跳变
        # [用户] z_std 0.02：TCP 只允许在 cube 质心及以上（质心以上满分，下方 2cm 内线性衰减）
        #   ——禁从下方接近/越过质心；贴面（TCP 质心下方 3cm）→ 0
        params={"palm_dist_threshold": 0.015, "lin_vel_threshold": 0.10, "ang_vel_threshold": 0.10, "d_std": 0.01, "z_std": 0.02},
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
    # object_out_of_bounds = DoneTerm(       # 物体被推飞即终止
    #     func=mdp.object_out_of_bounds,
    #     params={"threshold": 1.0, "use_xy_only": True},
    # )
    # object_away_from_robot = DoneTerm(     # 物体远离机器人即终止
    #     func=mdp.object_away_from_robot,
    #     params={"threshold": 2.0},
    # )
    # [已禁用 2026-08-31] debug_explosion 被策略利用作弊（reward hacking）：
    #   策略学会"甩臂→关节速度/位置爆表→触发本终止→episode 提前退出→规避 tcp_orientation 累计惩罚"。
    #   爆炸率 18%→50%（稳定），entropy 4.29→7.02、noise std 0.31→0.36（熵爆前兆），
    #   而 reach 反从 1.68→1.12 下降——"reward 转正"是假象（负惩罚被提前退出规避）。
    #   已解除本终止条件消除作弊通道；函数 mdp.debug_explosion 保留供后续排查数值爆炸根因。
    # debug_explosion = DoneTerm(func=mdp.debug_explosion)
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