import isaaclab.sim as sim_utils

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

DRILL_UR5E_CFG = ArticulationCfg(

    spawn=sim_utils.UsdFileCfg(

        usd_path="/home/jiangli/zhaoyucheng/assets/drill_ur5e_v1.usd",

        # v48: 恢复接触上报——用户要修传感器，让 env 知道"抓没抓紧"。
        # （之前 False 因为新系统上传感器恒为 0 + 每步开销大；现在配合 env_cfg 恢复 ContactSensorCfg）
        activate_contact_sensors=True,
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

            stiffness=40.0,
            damping=2.0,

        ),

    }

)