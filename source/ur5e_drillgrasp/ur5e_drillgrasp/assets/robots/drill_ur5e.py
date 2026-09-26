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
            # [2026-09-19 22:20 物理软化·干预2] contact_offset 2mm→5mm：接触在表面外更早生成
            #   → 减少"深穿透→求解挣扎→collection time 暴涨"（20:57 run 崩坏前兆）。
            #   代价：手指离表面 5mm 内即产生接触力（触觉半径变大，柔化接触）。
            contact_offset=0.005,
            rest_offset=0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            # [2026-09-19 22:20 物理软化·干预2] 穿透回弹速度 1.0→0.5：深穿透恢复更柔，减冲击。
            max_depenetration_velocity=0.5,
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
            "shoulder_lift_joint":-2.0944,   # -120°
            "elbow_joint":-2.2689,           # -130°
            "wrist_1_joint":-1.9199,         # -110°
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



            # [2026-09-20 预弯手型] 触发条件达成：v3 训练到 1871 iter，contact_gate 始终 0
            #   （接近阶段死锁）且物理崩坏 → 按预案加预弯，缩短"指尖→对置接触"的探索距离：
            #   中指/无名指（工作对）j2=0.45、j3=0.2、j4=0.2（半包络）；
            #   拇指 j1=-30°(-0.5236)、j2=-20°(-0.3491)（向对置方向预置）。
            #   方向依据：四指正角=弯曲；thumb1/2 负角=侧摆/屈曲（限位 [-π/2,0] / ±π/2）。
            #   ⚠️ 若 play 见手指反向翻出/穿 cube，调整符号或减小值。
            "thumb1_joint": -0.6981,  # -40°
            "thumb2_joint": -1.3962,  # -80°
            # [v25 回退3d] 保持出生即就位(+50°)：3d 改自然起点需从头训（动作基线位移
            #   使 resume 崩溃），留待接近阶段一起处理。回退目标仍是 0.0。
            "thumb3_joint": 0.8727,
            "thumb4_joint": 0.6981,   # +40°
            "index1_joint": -0.087,   # -5°（略张开防碰）
            "index2_joint": 1.0472,    # 60°
            "index3_joint": 0.5236,    # 30°
            "index4_joint": 0.0,       # 0°
            "middle1_joint": 0.0,
            # [2026-09-22 P1 试验记录·已回退] 曾把 middle3 0.4363→0.65、middle2 1.2217→1.20
            #   试图防穿模（几何扫描证明可行：指尖距面 0.65cm/中节间隙 +0.40cm），但动作是
            #   "相对默认角"的——改默认导致旧策略抓握失效（gate 0.94→0.47、达标率 23%→0%），
            #   用户决定回退预设、接受轻微穿模。若将来要再做，需配合一次策略适应训练。
            # [2026-09-22 用户定·路径延长] 中指由预弯(70°/25°)改伸展(j2=j3=20°)：
            #   "不预设中指贴近角，让它从伸展学到贴面→夹住→提起"。配合 surface margin 0.10 驱动。
            #   回退：1.2217 / 0.4363
            # [v19 用户决定] 20°→30°（0.3491→0.5236）：配合多段碰撞后的新接触几何
            #   （实测新碰撞下可接触姿态 m2≈1.1~1.4 / m3≈0.4~0.5，起点更近便于探索）。
            #   回退：0.3491。
            "middle2_joint": 0.5236,   # 30°
            "middle3_joint": 0.5236,   # 30°
            "middle4_joint": 0.0,      # 0°
            "ring1_joint": 0.0,
            "ring2_joint": 1.0472,     # 60°
            "ring3_joint": 0.5236,     # 30°
            "ring4_joint": 0.0,        # 0°
            "little1_joint": 0.0,
            "little2_joint": 0.8727,   # 50°
            "little3_joint": 0.5236,   # 30°
            "little4_joint": 0.0,      # 0°


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

            # [2026-09-19 22:30 干预6] stiffness 50→35、damping 6→5：接触刚度再降 30%，
            #   深压时的接触力峰值更柔（1258 尖峰→1279 崩的物理侧防线）。
            #   力上限由 effort（0.5 Nm）决定，仍 >10N 指尖力，不影响力封闭。
            #   回退：50 / 6.0。
            stiffness=35.0,
            # [2026-09-21 v5.2 抑颤] damping 5→8：实测关/接触时关节 1~2Hz 摆动
            #   （middle3 vel_rms 0.51 rad/s、接触力 std 53%）——提高物理阻尼压制。
            #   回退：5.0。
            damping=8.0,
            # [2026-09-21 v5 用户决定] 0.5→1.0 Nm：诊断实证——0.5 在接触下关节被卡死
            #   （middle3 指令 0.735 / 实际 0.11，err 0.63rad；thumb 中节顶住 cube 时同样跟不上），
            #   导致"指令在弯、实际被接触几何推直"的怪姿势。1.0 Nm ≈ 指尖 20N（需求 <1N），
            #   跟踪更好；若再现"深穿→求解挣扎/collection 暴涨"，回退 0.5（历史防深穿值）。
            # [v12.1 探针 2026-09-22] 1.0→3.0：确定性轨迹实证——damping 8 下关节最大
            #   速度 ≈ effort/damping = 0.125 rad/s，中指从伸展卷到参考(0.85rad)需 ~7s≈整集，
            #   策略"发起了也吃不到收益"→稳定解回伸直(弹簧平衡 m2≈0.45)。3.0 时 ~0.375rad/s，
            #   卷拢 ~2.3s。若接触抖动/力尖峰再现，回退 1.0（历史稳定值）。
            effort_limit_sim=3.0,
            # [2026-09-14 原始注释] URDF 自带 effort=10 Nm（对手指是巨值）：深穿透时位置控制
                                    #   持续"推" → 推穿力可达 10Nm 级 → 手指绞在物体/彼此之间 →
                                    #   PhysX 求解爆炸（崩坏时 collection 8s 的嫌疑机制）。

        ),

    }

)