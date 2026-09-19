import isaaclab.sim as sim_utils

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


DRILL_UR5E_CFG = ArticulationCfg(

    spawn=sim_utils.UsdFileCfg(

        # v68: 改回 v1 之前的原资产 drill_ur5e.usd（v61 尝试 v1 资产后用户要求改回）
        usd_path="/home/jiangli/zhaoyucheng/assets/drill_ur5e_v1.usd",

        # Stage 2: 恢复接触传感器（用户确认新系统传感器有数据）——指尖触觉 reward/观测需要
        activate_contact_sensors=True,
        # [2026-09-08] 注：UsdFileCfg 不支持 physics_material 字段，手指摩擦只能靠
        #   ①全局 SimulationCfg.physics_material（已设 μs=μd=1.0，USD 未自定义材质时生效）
        #   ②USD 文件内部物理材质（需在 USD 编辑器里确认手指碰撞体是否自带材质）
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.002,      # 2mm 接触容差
            rest_offset=0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=1.0,
        ),

        # [2026-09-14 防"绞死"] 显式关闭 articulation 内部自碰撞（指-指 / 指-掌 / 手-臂）：
        #   背景：崩坏时"乱动→手指深穿→PhysX 求解挣扎"会把 collection 从 ~3s 拉到 8s。
        #   影响范围：仅同一 articulation 内的 link 对；手指-cube、手指-地面等外部碰撞不受影响
        #   （collision_props 与 contact sensors 照常，手指-cube 接触/触觉/抓取奖励链全部保留）。
        #   代价：乱动/未收敛段可能出现视觉穿模（正常抓取段手指被 cube 分隔，大概率无碍）。
        #   后路：若 S3 验收发现成功段手指互插，可改回 True 做短微调对比。
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),

    ),

    init_state=ArticulationCfg.InitialStateCfg(

        pos=(0.0,0.0,0.0),

        joint_pos={

            # [2026-09-15 v2·缩距起点] 初始手掌位于 cube 上方约 20cm（原 ~40cm）——缩短"探索跨度"，
            #   降低 S1 起步阶段冻结概率（40cm 版曾冻在 21cm；起点直接设在 20cm，成功区更近）。
            #   角度来自用户 IsaacSim GUI 实测 [90,-100,-120,-145,-90,180]°（手掌朝下）→ 弧度：
            #   ⚠️ σ 地板/train.py 补丁仍建议后续补上——缩距治概率，地板治机制。
            "shoulder_pan_joint":1.5708,     # 90°
            "shoulder_lift_joint":-1.7453,   # -100°（原 -90°）
            "elbow_joint":-2.0944,           # -120°（原 -90°）
            # [2026-09-16 预科 v2·用户 GUI 实测] wrist_1 -145°→-140°：手拖姿势关键角之一
            #   （与 thumb2=-60°、thumb4=+40° 同组；thumb1/thumb3 保持 -10°/0°）
            "wrist_1_joint":-2.4435,         # -140°（原 -145°）
            "wrist_2_joint":-1.5708,         # -90°
            "wrist_3_joint":3.1416,          # 180°，手掌朝下

            # # 拇指初始位置：[用户 2026-09-01] thumb1 -30°→-10°：play 发现 -30° 时拇指朝下戳 cube，
            # #   悬停时拇指尖侵入 cube 顶面空间 → cube 被扰动 → success"稳定"条件不满足、
            # #   且拇指单侧戳无对向 → grip=0、finger_contact 无真实贴面。改平到 -10° 让拇指接近伸直。
            # #   注意 thumb3/4 下限=0，初始 0 正好贴下限边界（如需余量可调 +0.05）
            # "thumb1_joint":-0.175,   # -10°（限位 [-90°,0°] 内）
            #
            # "thumb2_joint":-1.0472,  # -60°（原 -20°）
            # "thumb3_joint":0.0,      # 下限0°（本次未指定，保持）
            # "thumb4_joint":0.6981,   # +40°（原 0°）（thumb4_hoint 已改名 thumb4_joint）
            #
            # # [2026-09-16 预弯手型·物理课程] 四指 j3/j4 出厂即半弯 20°（0.3491 rad）：
            # #   缩短"指尖→cube"探索距离（悬垂位指尖距表面仅 1~2cm；弯 j3/j4 会让指尖扫向 cube），
            # #   让指尖更容易真碰上 → finger_contact/finger_close 拿到正反馈起点。
            # #   保持半弯零成本：零动作 target=default 即此姿态；悬停区手动作罚被门控不生效。
            # #   方向依据：四指关节正角度=弯曲方向（与动作正输出同号；j2 实测 +1.0 rad≈弯 57°）。
            # #   ⚠️ 改完须同步训练机；若 play 里手指反向翻出，把 j3/j4 符号改负即可。
            # "index1_joint":0.0,    #-8度，手指略张开防碰
            # "index2_joint":0.50,       # 30° 左右预弯
            # "index3_joint":0.0,   # 10° 预弯
            # "index4_joint":0.0,   # 10° 预弯
            # "middle1_joint":0.0,
            # "middle2_joint":0.50,       # 30° 左右预弯
            # "middle3_joint":0.0,  # 10° 预弯
            # "middle4_joint":0.0,  # 10° 预弯
            # "ring1_joint":0.0,
            # "ring2_joint":0.50,       # 30° 左右 预弯
            # "ring3_joint":0.0,    # 10° 预弯
            # "ring4_joint":0.0,    # 10° 预弯
            # "little1_joint":0.0,  #8度，手指略张开防碰
            # "little2_joint":0.50,       # 30° 左右 预弯
            # "little3_joint":0.0,  # 10° 预弯
            # "little4_joint":0.0,  # 10° 预弯



            "thumb1_joint": 0,  # -10°（限位 [-90°,0°] 内）
            "thumb2_joint": 0,  # -60°（原 -20°）
            "thumb3_joint": 0.0,  # 下限0°（本次未指定，保持）
            "thumb4_joint": 0,  # +40°（原 0°）（thumb4_hoint 已改名 thumb4_joint）
            "index1_joint": 0.0,  # -8度，手指略张开防碰
            "index2_joint": 0,  # 30° 左右预弯
            "index3_joint": 0.0,  # 10° 预弯
            "index4_joint": 0.0,  # 10° 预弯
            "middle1_joint": 0.0,
            "middle2_joint": 0.0,  # 30° 左右预弯
            "middle3_joint": 0.0,  # 10° 预弯
            "middle4_joint": 0.0,  # 10° 预弯
            "ring1_joint": 0.0,
            "ring2_joint": 0.0,  # 30° 左右 预弯
            "ring3_joint": 0.0,  # 10° 预弯
            "ring4_joint": 0.0,  # 10° 预弯
            "little1_joint": 0.0,  # 8度，手指略张开防碰
            "little2_joint": 0.0,  # 30° 左右 预弯
            "little3_joint": 0.0,  # 10° 预弯
            "little4_joint": 0.0,  # 10° 预弯

        },

    ),

    actuators={

        "arm":ImplicitActuatorCfg(

            joint_names_expr=[

                "shoulder_pan_joint$",
                "shoulder_lift_joint$",
                "elbow_joint$",
                "wrist_1_joint$",
                "wrist_2_joint$",
                "wrist_3_joint$",

            ],

            stiffness=0.0,      # 纯力矩模式，OSC 直通
            damping=0.0,
            effort_limit_sim=300.0,   # 足够大，不截断重力补偿力矩
            velocity_limit_sim=10.0,  # 速度限位

        ),

        "hand":ImplicitActuatorCfg(

            joint_names_expr=[

                "thumb.*",
                "index.*",
                "middle.*",
                "ring.*",
                "little.*",

            ],

            stiffness=50.0,     # [2026-09-09] 80→50（50g 重训）：力=k×误差，刚度降→接触瞬态冲击力降→轻 cube 不被推飞。
                                #   50g 门槛 0.245N，刚度 50 绰绰有余（策略多弯补偿稳态力）。
                                #   历史：40→80 升力成功、80→90 升力崩；本次反向降力（匹配轻 cube）是安全方向。
                                # [2026-09-14 同步] 本机 80→50：与训练机实际值对齐（用户确认训练机=50）。
            damping=6.0,        # 随刚度 50：略过阻尼（临界 d≈2√(k·J)≈3.16），接触更柔、更平稳，防轻 cube 振荡
            effort_limit_sim=2.0,   # [2026-09-14 防"绞死"] URDF 自带 effort=10 Nm（对手指是巨值）：
                                    #   深穿透时位置控制持续"推" → 推穿力可达 10Nm 级 → 手指绞在
                                    #   物体/彼此之间 → PhysX 求解爆炸（崩坏时 collection 8s 的嫌疑机制）。
                                    #   正常操作需求 <1 Nm；2.0 留 2~4 倍余量，不影响抓握。

        ),

    }

)