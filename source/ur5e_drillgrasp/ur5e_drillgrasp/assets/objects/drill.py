import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


DRILL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Drill",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/jiangli/assets/drill.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            linear_damping=0.2,
            angular_damping=0.2,
            max_linear_velocity=20.0,
            max_angular_velocity=20.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),

    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.60, 0.00, 0.45),
        rot=(1.0,0.0,0.0,0.0),
    ),

)