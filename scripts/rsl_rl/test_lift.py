# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""判定实验：加载 model_4400，让策略正常抓取，然后强制手臂沿世界"上"方向抬起，
看 cube 是否被带起——回答"现有抓握物理上能不能举升"，把物理能力与奖励设计分开。"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Test whether the learned grasp can lift the cube.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--grasp_steps", type=int, default=240, help="策略自由运行步数，让抓握建立（控制频率 30Hz，默认 8s）"
)
parser.add_argument(
    "--lift_steps", type=int, default=120, help="强制上抬步数（默认 4s）"
)
parser.add_argument(
    "--lift_mag", type=float, default=5.0, help="上抬动作幅值（OSC position_scale=0.01，5.0→5cm 相对目标）"
)
parser.add_argument(
    "--lift_sign", type=float, default=1.0, help="上抬方向符号（若 palm 没上升则试 -1）"
)
parser.add_argument(
    "--freeze_hand", action="store_true", default=True,
    help="抬升阶段把手指动作冻结在抓握末态（隔离'抓握本身能否抱持'）。"
         "加 --no_freeze_hand 则手指仍由策略实时输出（测'策略是否在抬升中松手'）。",
)
parser.add_argument(
    "--no_freeze_hand", action="store_false", dest="freeze_hand",
    help="抬升阶段手指动作继续由策略输出（原行为）。",
)
parser.add_argument(
    "--grip_scale", type=float, default=4.0,
    help="抬升阶段把冻结的手部动作放大多少倍（默认4）。手是相对位置控制：目标=当前角+动作*0.05，"
         "策略末态动作≈0 → 零保持力矩、手指被动。放大后目标前移 → 产生真实夹持力矩。0=不放大。",
)
parser.add_argument(
    "--stiffness", type=float, default=None,
    help="临时覆盖手部执行器刚度（默认5.0）。相对控制下夹持力矩=刚度*动作*0.05，"
         "刚度不足=握力结构性不足。仅测试用，不改训练配置。",
)
parser.add_argument(
    "--hand_scale", type=float, default=None,
    help="临时覆盖 GroupedHandAction scale（默认0.05）。放大=每次目标偏移更大=力矩更大。"
         "仅测试用，会改变动作语义。",
)
parser.add_argument(
    "--max_squeeze", action="store_true",
    help="抬升阶段绝对握死：按策略学到的方向把每个手部关节推到限位（临时把 scale 顶到 2.0 使目标饱和，"
         "绕开 clip_actions 裁剪与相对控制'目标随动'的力矩上限）。配合 --stiffness 效果最佳。",
)
parser.add_argument(
    "--height_sweep", type=str, default=None,
    help="逗号分隔的手掌下移高度(cm)列表，如 '0,2,4'。每档：重置→重新抓握→手掌沿世界-Z下移→握死抬升，"
         "找'哪个手掌高度能形成可抱持的抓握'。",
)
parser.add_argument(
    "--cube_size", type=float, default=None,
    help="临时改变 cube 边长(cm)，如 5 = 5cm。仅测试用（不改训练配置），验证手能否包住不同尺寸的物体。",
)
parser.add_argument(
    "--cube_pos", type=str, default=None,
    help="cube 位置偏移(cm)，逗号分隔，相对原生位置(-0.18,0.11,0.785)。如 '0,0,0'=原位，'-1,2,0'=x-1cm,y+2cm。仅测试用。",
)
# append RSL-RL cli arguments (for --experiment_name / --load_run / --checkpoint)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply, quat_conjugate

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import ur5e_drillgrasp.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """加载 checkpoint，让策略抓取，再强制上抬，观察 cube 是否被带起。"""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.cube_size is not None:
        s = args_cli.cube_size / 100.0
        env_cfg.scene.cube_obj.spawn.size = (s, s, s)
        print(f"[INFO] cube size -> {s*100:.0f}cm")
    if args_cli.cube_pos is not None:
        dx, dy, dz = (float(v) for v in args_cli.cube_pos.split(","))
        _base = (-0.18, 0.11, 0.785)
        env_cfg.scene.cube_obj.init_state.pos = (_base[0] + dx / 100.0, _base[1] + dy / 100.0, _base[2] + dz / 100.0)
        print(f"[INFO] cube pos offset -> ({dx},{dy},{dz}) cm")

    # 定位 checkpoint：
    # - 传了绝对路径 → retrieve_file_path 直接用
    # - 传了 --load_run + --checkpoint(文件名) → get_checkpoint_path 拼路径（与 play.py 一致）
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_root_path = os.path.join(_project_root, "logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint and os.path.isabs(args_cli.checkpoint):
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    def _get_hand_term():
        """兼容 terms / _terms 两种访问方式，取手部动作 term。"""
        am = env.unwrapped.action_manager
        for attr in ("terms", "_terms"):
            d = getattr(am, attr, None)
            if isinstance(d, dict) and "hand_action" in d:
                return d["hand_action"]
        return None

    # ── 临时物理/控制覆盖（仅测试用，不改训练配置）──
    robot_tmp = env.unwrapped.scene["robot"]
    if args_cli.stiffness is not None:
        # 1) 走 actuator 属性（部分版本 setter 只存值不推物理，所以再走物理视图兜底）
        try:
            act = env.unwrapped.scene.articulations["robot"].actuators["hand"]
            act.stiffness = args_cli.stiffness
            print(f"[INFO] hand actuator stiffness (actuator) -> {act.stiffness}")
        except Exception as e:
            print(f"[WARN] actuator stiffness 设置失败: {e}")
        # 2) 物理视图直接设置（绕过 setter 是否生效的不确定性）
        try:
            hand_ids, _ = robot_tmp.find_joints("(thumb|index|middle|ring|little).*")
            stiff = torch.full((len(hand_ids),), args_cli.stiffness, device=robot_tmp.device)
            robot_tmp._physics_view.set_joint_drive_stiffness(stiff, joint_ids=hand_ids)
            print(f"[INFO] hand joint drive stiffness set via physics_view (n={len(hand_ids)})")
        except Exception as e:
            print(f"[WARN] physics_view 设置 stiffness 失败: {e}")
    if args_cli.hand_scale is not None:
        term = _get_hand_term()
        if term is not None:
            term.cfg.scale = args_cli.hand_scale
            print(f"[INFO] GroupedHandAction scale -> {term.cfg.scale}")
        else:
            print("[WARN] 找不到 hand_action term，--hand_scale 未生效")

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] RslRlVecEnvWrapper clip_actions = {getattr(env, 'clip_actions', agent_cfg.clip_actions)}")

    # load policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 场景引用
    robot = env.unwrapped.scene["robot"]
    cube = env.unwrapped.scene["cube_obj"]
    body_idx, _ = robot.find_bodies("base_link_1")
    N = env.unwrapped.num_envs
    device = env.unwrapped.device

    # 世界"上"方向 → base_link_1 局部系
    world_up = torch.zeros(N, 3, device=device)
    world_up[:, 2] = 1.0

    # 手指状态摘要：每组第3关节(中段 m) + 第4关节(指尖 t)，用于诊断抬升中是否松手
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

    def log(msg):
        cz = cube.data.root_pos_w[:, 2].mean().item()
        pz = robot.data.body_pos_w[:, body_idx, 2].mean().item()
        print(f"[LIFT-TEST] {msg}  cube_z={cz:.4f}  palm_z={pz:.4f}  hand[{hand_summary()}]")

    # ── 阶段 1：策略自由运行，建立抓握 ──
    obs = env.get_observations()
    log("阶段0 初始")
    with torch.inference_mode():
        for t in range(args_cli.grasp_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
    cube_z_grasp = cube.data.root_pos_w[:, 2].mean().item()
    palm_z_grasp = robot.data.body_pos_w[:, body_idx, 2].mean().item()
    hand_action = actions[:, 6:].clone()   # 冻结源：策略最后一次输出的手指动作 (12D)
    action_dim = actions.shape[1]          # 应为 18 (arm 6 + hand 12)
    print(f"[LIFT-TEST] 冻结手部动作: mean={hand_action.mean().item():+.4f}  max|.|={hand_action.abs().max().item():.4f}  "
          f"(tanh 上限 1.0，目标偏移=动作*0.05，若≈0 则保持力矩≈0)")
    log("阶段1 抓握建立后")

    # ── 阶段 2：抬臂测试 ──
    # 绝对握死准备：scale 顶到 2.0（一次即可），squeeze 按策略学到的方向推满
    hand_term = _get_hand_term()
    if hand_term is not None:
        hand_term.cfg.scale = 2.0
        print(f"[INFO] hand term scale -> {hand_term.cfg.scale}")
    squeeze = torch.sign(hand_action) * 50.0
    squeeze[squeeze == 0] = 50.0
    print(f"[LIFT-TEST] 冻结 hand_action 12D = {[round(v,3) for v in hand_action[0].tolist()]}")
    print(f"[LIFT-TEST] squeeze 12D = {[round(v,1) for v in squeeze[0].tolist()]}")

    def _step(actions):
        nonlocal obs
        obs, _, dones, _ = env.step(actions)
        if version.parse(installed_version) >= version.parse("4.0.0"):
            policy.reset(dones)

    def _step_lift(sq, steps, mag):
        """握死 + 抬臂 steps 步，返回最终 (cube_z, palm_z)。"""
        with torch.inference_mode():
            for _ in range(steps):
                quat = robot.data.body_quat_w[:, body_idx]
                up_body = quat_apply(quat_conjugate(quat), world_up)
                actions = torch.zeros(N, action_dim, device=device)
                actions[:, :3] = up_body * mag * args_cli.lift_sign
                actions[:, 6:] = sq
                _step(actions)
        return (cube.data.root_pos_w[:, 2].mean().item(),
                robot.data.body_pos_w[:, body_idx, 2].mean().item())

    def _descend(sq, h, max_steps=60):
        """握死 + 沿世界 -Z 下移，直到 palm_z 下降 h 米（最多 max_steps 步）。"""
        p0 = robot.data.body_pos_w[:, body_idx, 2].mean().item()
        with torch.inference_mode():
            for _ in range(max_steps):
                quat = robot.data.body_quat_w[:, body_idx]
                down_body = quat_apply(quat_conjugate(quat), -world_up)
                actions = torch.zeros(N, action_dim, device=device)
                actions[:, :3] = down_body * 5.0
                actions[:, 6:] = sq
                _step(actions)
                pz = robot.data.body_pos_w[:, body_idx, 2].mean().item()
                if p0 - pz >= h:
                    break
        # 停住（零动作保持 8 步，让手稳定）
        with torch.inference_mode():
            for _ in range(8):
                actions = torch.zeros(N, action_dim, device=device)
                actions[:, 6:] = sq
                _step(actions)

    if args_cli.height_sweep is not None:
        # ── 高度扫描：每档重抓握 → 下移 h → 握死抬升，找能抱持的高度 ──
        heights = [float(x) for x in args_cli.height_sweep.split(",")]
        print(f"[LIFT-TEST] 高度扫描: 手掌相对抓握位下移 {heights} cm，每档重新抓握")
        for h in heights:
            with torch.inference_mode():
                obs, _ = env.reset()
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(torch.ones(N, device=device, dtype=torch.bool))
                for _ in range(args_cli.grasp_steps):
                    actions = policy(obs)
                    _step(actions)
            h_action = actions[:, 6:].clone()
            sq = torch.sign(h_action) * 50.0
            sq[sq == 0] = 50.0
            c0 = cube.data.root_pos_w[:, 2].mean().item()
            if h > 0:
                _descend(sq, h / 100.0)
            c_end, p_end = _step_lift(sq, args_cli.lift_steps, args_cli.lift_mag)
            print(f"[LIFT-TEST] h={h:.0f}cm  cube 升幅={c_end-c0:+.4f} m  palm_z_end={p_end:.4f}")
    else:
        # ── 单次抬升（freeze_hand / max_squeeze）──
        with torch.inference_mode():
            for t in range(args_cli.lift_steps):
                quat = robot.data.body_quat_w[:, body_idx]
                up_body = quat_apply(quat_conjugate(quat), world_up)
                if args_cli.freeze_hand:
                    actions = torch.zeros(N, action_dim, device=device)
                    actions[:, :3] = up_body * args_cli.lift_mag * args_cli.lift_sign
                    if args_cli.max_squeeze:
                        actions[:, 6:] = squeeze
                    else:
                        actions[:, 6:] = hand_action * args_cli.grip_scale
                else:
                    actions = policy(obs)
                    actions[:, :3] = up_body * args_cli.lift_mag * args_cli.lift_sign
                _step(actions)
                if t % 30 == 0 or t == args_cli.lift_steps - 1:
                    log(f"阶段2 t={t}")

        # ── 判定（阶段2结束 vs 阶段1结束）──
        cube_z_end = cube.data.root_pos_w[:, 2].mean().item()
        palm_z_end = robot.data.body_pos_w[:, body_idx, 2].mean().item()
        d_palm = palm_z_end - palm_z_grasp
        d_cube = cube_z_end - cube_z_grasp
        print("=" * 64)
        if args_cli.freeze_hand:
            mode = f"手指冻结×{args_cli.grip_scale}(freeze_hand)"
        else:
            mode = "手指策略控制(no_freeze_hand)"
        print(f"[LIFT-TEST] 模式: {mode}")
        print(f"[LIFT-TEST] 抓握建立时:  cube_z={cube_z_grasp:.4f}  palm_z={palm_z_grasp:.4f}")
        print(f"[LIFT-TEST] 上抬结束后:  cube_z={cube_z_end:.4f}  palm_z={palm_z_end:.4f}")
        print(f"[LIFT-TEST] palm 升幅 = {d_palm:+.4f} m")
        print(f"[LIFT-TEST] cube 升幅 = {d_cube:+.4f} m")
        print("-" * 64)
        if d_palm < 0.005:
            print("[LIFT-TEST] ❌ palm 没上升：抬臂没生效，试 --lift_sign -1 或加大 --lift_mag 重跑")
        elif d_cube > 0.01:
            if args_cli.freeze_hand:
                print("[LIFT-TEST] ✅ 冻结手指时能带起 cube——抓握物理上可举；")
                print("[LIFT-TEST]    之前 cube 不动很可能是策略抬升中松手 → Stage 3 重点=保持握持（冻结/固定手部动作），而非加大握力")
            else:
                print("[LIFT-TEST] ✅ 抓握能带起 cube（升幅 >=1cm）——物理上可举，问题在奖励设计，继续调 lift")
        else:
            if args_cli.freeze_hand:
                print("[LIFT-TEST] ❌ 手指冻结仍带不起 cube——抓握物理上抱不住")
                if not args_cli.max_squeeze and args_cli.stiffness is None and args_cli.hand_scale is None:
                    print("[LIFT-TEST]    下一步：加 --max_squeeze --stiffness 50 重跑（绝对握死+高刚度），拆解'力不够 vs 姿态不行'")
                elif not args_cli.max_squeeze:
                    print("[LIFT-TEST]    已加大动作/刚度仍不举（可能被 clip_actions=±1 裁剪抵消）→ 再加 --max_squeeze 重跑")
                else:
                    print("[LIFT-TEST]    已绝对握死+高刚度仍不举 → 纯姿态/接触几何问题（食中指不包络），需改手部几何/任务几何")
            else:
                print("[LIFT-TEST] ❌ palm 上升但 cube 没被带起——先加 --freeze_hand 重跑，区分'策略松手' vs '抓握本身抱不住'")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
