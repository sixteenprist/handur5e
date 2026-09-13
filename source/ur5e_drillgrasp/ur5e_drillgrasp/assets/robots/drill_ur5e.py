import isaaclab.sim as sim_utils

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg


DRILL_UR5E_CFG = ArticulationCfg(

    spawn=sim_utils.UsdFileCfg(

        # v68: 改回 v1 之前的原资产 drill_ur5e.usd（v61 尝试 v1 资产后用户要求改回）
        usd_path="/home/jiangli/zhaoyucheng/assets/drill_ur5e_v11.usd",

        # Stage 2: 恢复接触传感器（用户确认新系统传感器有数据）——指尖触觉 reward/观测需要
        activate_contact_sensors=True,
        # [2026-09-08] 注：UsdFileCfg 不支持 physics_material 字段，手指摩擦只能靠
        #   ①全局 SimulationCfg.physics_material（已设 μs=μd=1.0，USD 未自定义材质时生效）
        #   ②USD 文件内部物理材质（需在 USD 编辑器里确认手指碰撞体是否自带材质）
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.002,  # 2mm 接触容差
            rest_offset=0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=1.0,
        ),

    ),

    init_state=ArticulationCfg.InitialStateCfg(

        pos=(0.0,0.0,0.0),

        joint_pos={

            "shoulder_pan_joint":1.5708,
            "shoulder_lift_joint":-1.5708,
            "elbow_joint":-1.5708,
            "wrist_1_joint":-3.1416,
            "wrist_2_joint":-1.5708,
            "wrist_3_joint":3.1416,         # 手掌朝下

            # 拇指初始位置：[用户 2026-09-01] thumb1 -30°→-10°：play 发现 -30° 时拇指朝下戳 cube，
            #   悬停时拇指尖侵入 cube 顶面空间 → cube 被扰动 → success"稳定"条件不满足、
            #   且拇指单侧戳无对向 → grip=0、finger_contact 无真实贴面。改平到 -10° 让拇指接近伸直。
            #   注意 thumb3/4 下限=0，初始 0 正好贴下限边界（如需余量可调 +0.05）
            "thumb1_joint":-0.175,   # -10°（限位 [-90°,0°] 内）
            "thumb2_joint":0.0,      # 限位 [-60°,60°]，0° 在中间
            "thumb3_joint":0.0,      # 下限0°
            "thumb4_joint":0.0,      # 下限0°（用户已在 USD 里把 thumb4_hoint 改成 thumb4_joint）
            "index.*_joint":0.0,
            "middle.*_joint":0.0,
            "ring.*_joint":0.0,
            "little.*_joint":0.0,

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

            stiffness=80.0,     # [2026-09-06] 40→80：Stage 3 举升力不足（opp≈0.06N vs 夹起 200g 需 0.98N）。
                                #   力=k×位置误差，刚度×2 让同样弯曲产出 ×2 力（0.06→0.12N），且 grip=tanh(opp/0.05)
                                #   对同样动作给更高分→策略有梯度跟着加压。200g 静摩擦 1.96N 足够，不会推飞（推飞是 100g 的问题）。
            damping=6.0,        # 随刚度 80：临界阻尼 d≈2√(k·J)，k×2 → d×2（2.0→4.0），防欠阻尼接触力振荡

        ),

    }

)