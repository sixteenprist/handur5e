"""诊断指尖接触传感器——加载策略真实抓取 cube，抓取建立后读传感器力。

用法（远程）:
  ./isaaclab.sh -p ~/zhaoyucheng/RobotProject/ur5e_drillgrasp/scripts/rsl_rl/test_sensors.py \
    --task Template-Ur5e-Drillgrasp-v0 --num_envs 1 \
    --load_run <时间戳> --checkpoint model_XXXX.pt

先让策略正常抓取（--grasp_steps 步），抓取建立后每 30 步打印各指尖力。
若全部为 0 → 传感器/USD 碰撞问题；若非 0 → 传感器正常，是读取代码或训练配置问题。
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="诊断指尖接触传感器（加载策略真实抓取）")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--grasp_steps", type=int, default=300, help="策略自由运行步数，让抓握建立（控制频率 30Hz，默认 10s）"
)
# append RSL-RL cli arguments (for --experiment_name / --load_run / --checkpoint)
import cli_args  # isort: skip
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402
from packaging import version  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import ur5e_drillgrasp.tasks  # noqa: F401

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 定位 checkpoint（与 test_lift.py 一致）
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_root_path = os.path.join(_project_root, "logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint and os.path.isabs(args_cli.checkpoint):
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    cube = env.unwrapped.scene["cube_obj"]
    body_idx, _ = robot.find_bodies("base_link_1")
    N = env.unwrapped.num_envs
    device = env.unwrapped.device

    # ── 0. 碰撞体检查：遍历 USD 看每个指尖 link 是否有 collision prim ──
    print("=" * 60)
    print("[DIAG] 碰撞体检查（遍历 USD stage）...")
    try:
        from pxr import Usd, UsdShade, UsdGeom
        stage = simulation_app.context.get_stage()
        for grp in ("thumb", "index", "middle", "ring", "little"):
            link_name = f"{grp}4"
            # 通过 scene 找到 robot 的 root prim path
            root_path = env.unwrapped.scene["robot"].cfg.prim_path
            # 枚举该 link 下的子 prim，找 collision 相关
            link_prim_path = f"{root_path.replace('{ENV_REGEX_NS}', '/World/envs/env_0')}/hand/{link_name}"
            link_prim = stage.GetPrimAtPath(link_prim_path)
            if not link_prim.IsValid():
                print(f"[DIAG] {link_name}: ❌ prim 不存在于 {link_prim_path}")
                continue
            has_collision = False
            for child in link_prim.GetChildren():
                if "collision" in child.GetName().lower() or "collider" in child.GetName().lower():
                    has_collision = True
                    print(f"[DIAG] {link_name}: ✅ 找到 collision prim: {child.GetPath()}")
            if not has_collision:
                # 也许 link 本身有 collision API
                if link_prim.HasAPI("PhysxSchema.PhysxCollisionAPI"):
                    print(f"[DIAG] {link_name}: ✅ link 自带 PhysxCollisionAPI")
                else:
                    print(f"[DIAG] {link_name}: ❌ 未找到 collision prim！(子节点: {[c.GetName() for c in link_prim.GetChildren()][:10]})")
    except Exception as e:
        print(f"[DIAG] 碰撞体检查失败: {type(e).__name__}: {e}")

    sensor_names = ["contact_thumb", "contact_index", "contact_middle", "contact_ring", "contact_little"]

    def read_forces():
        """读五个指尖的接触力范数 → (N, 5)。"""
        forces = torch.zeros(N, 5, device=device)
        for i, name in enumerate(sensor_names):
            sensor = env.unwrapped.scene.sensors[name]
            fmat = sensor.data.force_matrix_w
            if fmat is not None and fmat.numel() > 0:
                forces[:, i] = torch.norm(fmat[:, 0, 0, :], dim=-1)
        return forces

    def hand_summary() -> str:
        jp = robot.data.joint_pos
        parts = []
        for grp in ("thumb", "index", "middle", "ring", "little"):
            j, _ = robot.find_joints(grp + ".*")
            if len(j) > 3:
                mid = jp[:, j[2]].mean().item()
                tip = jp[:, j[3]].mean().item()
                parts.append(f"{grp[:3]}(m{mid:.2f}/t{tip:.2f})")
        return "  ".join(parts)

    obs = env.get_observations()
    cz = cube.data.root_pos_w[:, 2].mean().item()
    pz = robot.data.body_pos_w[:, body_idx, 2].mean().item()
    print(f"[DIAG] 阶段0 初始  cube_z={cz:.4f}  palm_z={pz:.4f}  hand[{hand_summary()}]")

    with torch.inference_mode():
        for t in range(args_cli.grasp_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            if t % 30 == 0 or t == args_cli.grasp_steps - 1:
                f = read_forces()
                fmax = f.max(dim=-1).values.mean().item()
                cz = cube.data.root_pos_w[:, 2].mean().item()
                pz = robot.data.body_pos_w[:, body_idx, 2].mean().item()
                print(f"[DIAG] step={t:3d}  cube_z={cz:.4f} palm_z={pz:.4f}  "
                      f"max_force={fmax:.4f}N  hand[{hand_summary()}]")
                if fmax > 0:
                    for i, name in enumerate(sensor_names):
                        print(f"        {name}: {f[:, i].mean().item():.4f}N")

    f = read_forces()
    fmax = f.max(dim=-1).values.mean().item()
    print("=" * 60)
    if fmax > 0:
        print(f"[DIAG] ✅ 传感器有读数！最大力={fmax:.4f}N —— 传感器正常，contact_force=0 是训练/配置问题")
        for i, name in enumerate(sensor_names):
            print(f"[DIAG]   {name}: {f[:, i].mean().item():.4f}N")
    else:
        print(f"[DIAG] ❌ 抓握建立后力仍全 0 —— 传感器/USD 碰撞问题，不是读取代码问题")
        print(f"[DIAG]    检查: 1) USD 里指尖 link 是否有碰撞体 2) filter_prim_paths_expr 是否匹配 cube")
    env.close()


if __name__ == "__main__":
    main()
