# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""最终策略验收评测：N 集确定性 rollout（自动课程"末态=最终部署配置"）。

用途
----
加载训练权重（如自动课程清跑产物 ``best/grasp_v49_newgeom_autofinal.pt``），
在**最终部署配置**下跑 N 集确定性 rollout（策略取均值、无探索噪声），统计：

  - 成功率      : 提起净高度 ≥ 8cm 的集数占比（8cm 与 ``lift_success_reward`` 一致）
  - 提起高度    : 成功集的最大净提升（cm）
  - 抓取瞬间力  : 首次跨过 8cm 那一步的 拇指力 / 四指最大力（N）
  - 抓取瞬间漂移: 同一时刻的 cube 水平位移（cm）——"抓取有没有把 cube 推跑"
  - 全程漂移    : 整集 cube 水平位移最大值（cm，含提起后保持期）

为什么必须用"末态配置"
----------------------
``ur5e_drillgrasp_env_cfg.py`` 里存的是自动课程**起始易值**（预备位/预闭合、松门控、
tip_progress=12…）；直接用会与已训练策略分布错配（历史实测成功率骤降）。
本脚本调用 ``apply_final_deploy_overrides()`` 一键切到课程末态：
curriculum 关闭、远起 scale=1.0、预备位/预闭合=0、门控链终值、t2 自由。

用法（项目根目录）
------------------
    python scripts/rsl_rl/eval_final.py \
        --checkpoint logs/rsl_rl/ur5e_grasp_v5/best/grasp_v49_newgeom_autofinal.pt \
        --num_envs 100 --steps 300

参考验收线（2026-09-30，v49 新几何"从上往下"清跑产物）
--------------------------------------------------------
    成功率 100/100；提起 mean 10.6cm / min 9.8cm；
    抓取瞬间 拇 0.65N / 四指max 0.26N；抓取瞬间漂移 0.0cm；
    全程漂移 mean 3.1 / max 4.0cm（提起后保持期，属可接受范围）。
"""

import argparse

# ── 解析命令行后再起 Isaac 应用 ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="最终策略验收评测（N 集确定性 rollout）")
parser.add_argument("--task", type=str, default="Template-Ur5e-Drillgrasp-v0", help="任务名")
parser.add_argument("--checkpoint", type=str, required=True, help="权重文件路径（.pt）")
parser.add_argument("--num_envs", type=int, default=100, help="评测环境数 = 集数（默认 100）")
parser.add_argument("--steps", type=int, default=300, help="每集步数（默认 300 ≈ 5s）")
args_cli = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True})
sim = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401
import ur5e_drillgrasp.tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from ur5e_drillgrasp.tasks.manager_based.ur5e_drillgrasp.agents.rsl_rl_ppo_cfg import PPORunnerCfg  # noqa: E402
from ur5e_drillgrasp.tasks.manager_based.ur5e_drillgrasp.mdp import rewards as R  # noqa: E402
from ur5e_drillgrasp.tasks.manager_based.ur5e_drillgrasp.ur5e_drillgrasp_env_cfg import (  # noqa: E402
    apply_final_deploy_overrides,
)

SUCCESS_HEIGHT = 0.08  # 提起成功阈值（m）：与 lift_success_reward(max_height=0.08) 对齐


def main():
    N, T = args_cli.num_envs, args_cli.steps

    # 环境：直接构建 + 一键切到"最终部署配置"（关键！见文件头说明）
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=N)
    apply_final_deploy_overrides(cfg)
    env = gym.make(args_cli.task, cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # 策略：确定性（act_inference 取均值）
    runner = OnPolicyRunner(env, PPORunnerCfg().to_dict(), log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uw = env.unwrapped
    cube = uw.scene["cube_obj"]
    obs, _ = env.reset()
    cp0 = cube.data.root_pos_w.clone()  # 每集起始 cube 位置（reset_cube 精确复位）

    zmax = torch.zeros(N, device=uw.device)  # 净提升最大值
    dmax = torch.zeros(N, device=uw.device)  # 全程 xy 漂移最大值
    d_at = torch.full((N,), float("nan"), device=uw.device)  # 首达 8cm 时刻的 xy 漂移
    f_at = torch.zeros(N, device=uw.device)  # 首达 8cm 时刻的拇指力
    f4_at = torch.zeros(N, device=uw.device)  # 首达 8cm 时刻的四指最大力

    with torch.no_grad():
        for _ in range(T):
            a = policy(obs)
            obs, _, _, _ = env.step(a)
            fv = torch.norm(R._fingertip_force_vectors(uw), dim=-1)  # (N,5) 指尖力
            dz = cube.data.root_pos_w[:, 2] - cp0[:, 2]
            dxy = torch.norm((cube.data.root_pos_w - cp0)[:, :2], dim=-1)
            newly = (zmax < SUCCESS_HEIGHT) & (dz >= SUCCESS_HEIGHT)  # 本步首次达标
            if newly.any():
                d_at[newly] = dxy[newly]
                f_at[newly] = fv[newly, 0]
                f4_at[newly] = fv[newly, 1:].max(dim=1).values
            zmax = torch.maximum(zmax, dz)
            dmax = torch.maximum(dmax, dxy)

    lift = zmax * 100
    dr = dmax * 100
    succ = zmax >= SUCCESS_HEIGHT

    print("=" * 64, flush=True)
    print(f"最终验收评测（N={N}，每集 {T} 步，确定性策略，最终部署配置）", flush=True)
    print(f"  权重: {args_cli.checkpoint}", flush=True)
    print("=" * 64, flush=True)
    print(f"  成功率      : {int(succ.sum())}/{N} = {float(succ.float().mean()) * 100:.1f}%", flush=True)
    if succ.any():
        print(
            f"  提起高度    : mean {float(lift[succ].mean()):.1f}cm / min {float(lift[succ].min()):.1f}cm",
            flush=True,
        )
        print(
            f"  抓取瞬间力  : 拇 {float(f_at[succ].mean()):.2f}N（min {float(f_at[succ].min()):.2f}）"
            f" / 四指max {float(f4_at[succ].mean()):.2f}N（min {float(f4_at[succ].min()):.2f}）",
            flush=True,
        )
        print(
            f"  抓取瞬间漂移: mean {float(d_at[succ].mean()):.1f}cm / max {float(d_at[succ].max()):.1f}cm"
            "（cube 未被推跑=好）",
            flush=True,
        )
    print(
        f"  全程漂移    : mean {float(dr.mean()):.1f}cm / max {float(dr.max()):.1f}cm"
        "（含提起后保持期）",
        flush=True,
    )
    env.close()
    sim.close()


if __name__ == "__main__":
    main()
