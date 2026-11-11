"""[v60 蒸馏] 收集专家策略的 (obs, action) 对，用于形状条件学生网络的 BC。

- cube 专家：v58（非池方块，111 维观测）
- ball 专家：球专用延迟鲁棒（OBJECT_KIND=sphere，111 维观测）
- 保存时在 obs 末尾追加形状位（cube=0 / ball=1）→ (M,112)；act (M,22)。

用法：
  python scripts/analysis/collect_distill_data.py --checkpoint <ckpt> --shape cube \
      --out logs/distill/cube.pt --num_envs 200 --steps 300
  python scripts/analysis/collect_distill_data.py --checkpoint <ball ckpt> --shape ball \
      --out logs/distill/ball.pt --num_envs 200 --steps 300
"""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="收集专家 BC 数据（方块/球）")
parser.add_argument("--task", type=str, default="Template-Ur5e-Drillgrasp-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--shape", type=str, choices=["cube", "ball"], required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=200)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--radius", type=float, default=0.03)
parser.add_argument("--seed", type=int, default=42)
args_cli = parser.parse_args()

# 专家使用 111 维单形状环境（形状位只在学生 env 打开）
os.environ.pop("SHAPE_OBS", None)
os.environ.pop("OBJECT_POOL", None)
if args_cli.shape == "ball":
    os.environ["OBJECT_KIND"] = "sphere"
    os.environ["OBJECT_SIZE"] = str(args_cli.radius)
else:
    os.environ["OBJECT_KIND"] = "box"

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
from ur5e_drillgrasp.tasks.manager_based.ur5e_drillgrasp.ur5e_drillgrasp_env_cfg import (  # noqa: E402
    apply_final_deploy_overrides,
)


def main():
    N, T = args_cli.num_envs, args_cli.steps
    torch.manual_seed(args_cli.seed)
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=N)
    cfg.seed = args_cli.seed
    apply_final_deploy_overrides(cfg)
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=cfg))
    runner = OnPolicyRunner(env, PPORunnerCfg().to_dict(), log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.reset()
    obs_buf, act_buf = [], []
    with torch.no_grad():
        for _ in range(T):
            o = obs["policy"].detach().clone()
            a = policy(obs)
            obs_buf.append(o.cpu())
            act_buf.append(a.detach().cpu())
            obs, _, _, _ = env.step(a)
    obs111 = torch.cat(obs_buf, dim=0)
    act = torch.cat(act_buf, dim=0)
    bit = torch.full((obs111.shape[0], 1), 1.0 if args_cli.shape == "ball" else 0.0)
    data = {"obs": torch.cat([obs111, bit], dim=1), "act": act, "shape": args_cli.shape,
            "checkpoint": args_cli.checkpoint}
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    torch.save(data, args_cli.out)
    print(f"[collect] shape={args_cli.shape} samples={data['obs'].shape[0]} "
          f"obs={tuple(data['obs'].shape)} act={tuple(data['act'].shape)} -> {args_cli.out}", flush=True)
    env.close()
    sim.close()


if __name__ == "__main__":
    main()
