import isaaclab.sim as sim_utils

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


DRILL_UR5E_CFG = ArticulationCfg(

    spawn=sim_utils.UsdFileCfg(

        usd_path="/home/jiangli/zhaoyucheng/assets/drill_ur5e.usd",
        # activate_contact_sensors=True,
        # 提速：禁用全关节接触上报（5 个指尖传感器已注释，新系统上恒为 0）；
        # 后续修好接触传感器/需要触觉奖励时改回 True 并恢复传感器
        activate_contact_sensors=False,
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

            # 拇指各关节限位方向不同，需分别设初值，避开限位边界
            "thumb1_joint":-0.3,     # 上限0，设-0.3避开
            "thumb2_joint":-0.3,
            "thumb3_joint":0.3,      # 下限0，设+0.3避开
            "thumb4_joint":0.3,      # 下限0，设+0.3避开
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

            stiffness=5.0,     # 中等刚度，ωn≈3.16 rad/s
            damping=0.5,       # 过阻尼 ζ≈1.58，绝不振荡

        ),

    }

)