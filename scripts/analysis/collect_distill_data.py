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
parser.add_argument("--augment", type=int, default=0, help="1=按部署扰动场景采集（延迟/滤波/限幅/噪声）")
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


# ── 与 eval_deploy.DeployShim 完全相同的扰动顺序：观测延迟/噪声 → 策略 → 低通 → 限幅 → 动作延迟 ──
import collections  # noqa: E402

N_POS, N_VEL = 26, 26  # obs 前 52 维 = 关节位置 26 + 关节速度 26


class DeployShim:
    def __init__(self, policy, beta=None, delay=0, rate_arm=None, noise_pos=0.0, noise_vel=0.0, act_delay=0):
        self.policy, self.beta, self.delay, self.rate_arm = policy, beta, int(delay), rate_arm
        self.act_delay, self.noise_pos, self.noise_vel = int(act_delay), float(noise_pos), float(noise_vel)
        self._act_buf, self._hist, self._prev = [], None, None
        self.debug_raw_action = None
        self.debug_obs_use = None

    def reset(self):
        self._hist = collections.deque(maxlen=self.delay + 1) if self.delay > 0 else None
        self._act_buf, self._prev = [], None

    def __call__(self, obs):
        t = obs["policy"]
        if self._hist is not None:
            self._hist.append(t.detach().clone())
            obs_use = self._hist[0]
        else:
            obs_use = t
        if self.noise_pos > 0.0 or self.noise_vel > 0.0:
            obs_use = obs_use.clone()
            if self.noise_pos > 0.0:
                obs_use[:, 0:N_POS] += torch.randn_like(obs_use[:, 0:N_POS]) * self.noise_pos
            if self.noise_vel > 0.0:
                obs_use[:, N_POS:N_POS + N_VEL] += torch.randn_like(obs_use[:, N_POS:N_POS + N_VEL]) * self.noise_vel
        self.debug_obs_use = obs_use.detach().clone()
        if self._hist is not None or self.noise_pos > 0.0 or self.noise_vel > 0.0:
            obs_td = obs.clone()
            obs_td["policy"] = obs_use
            a = self.policy(obs_td)
        else:
            a = self.policy(obs)
        self.debug_raw_action = a.detach().clone()  # BC 目标=滤波/限幅/延迟之前的专家动作
        if self._prev is None:
            self._prev = a.detach().clone()
        if self.beta is not None and self.beta < 1.0:
            a = self.beta * a + (1.0 - self.beta) * self._prev
        if self.rate_arm is not None:
            a = a.clone()
            d = (a[:, :6] - self._prev[:, :6]).clamp(-self.rate_arm, self.rate_arm)
            a[:, :6] = self._prev[:, :6] + d
        if self.act_delay > 0:
            self._act_buf.append(a.detach().clone())
            if len(self._act_buf) <= self.act_delay:
                a = torch.zeros_like(a)
            else:
                a = self._act_buf.pop(0)
        self._prev = a.detach().clone()
        return a


AUG_SCENARIOS = {
    "clean": {},
    "delay1": {"delay": 1},
    "delay2": {"delay": 2},
    "delay4": {"delay": 4},
    "beta0.5": {"beta": 0.5},
    "beta0.33": {"beta": 0.33},
    "beta0.5_d1": {"beta": 0.5, "delay": 1},
    "beta0.5_d2_noise": {"beta": 0.5, "delay": 2, "noise_pos": 0.002, "noise_vel": 0.01},
    "beta0.5_d1_noise": {"beta": 0.5, "delay": 1, "noise_pos": 0.002, "noise_vel": 0.01},
    "act1": {"act_delay": 1},
    "act2": {"act_delay": 2},
    "act1_b0.5": {"act_delay": 1, "beta": 0.5},
    "act1_obs1": {"act_delay": 1, "delay": 1},
    "rate1.5": {"rate_arm": 1.5},
    "rate2.0": {"rate_arm": 2.0},
}


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

    obs_buf, act_buf = [], []
    if args_cli.augment:
        with torch.no_grad():
            for i, (name, params) in enumerate(AUG_SCENARIOS.items()):
                torch.manual_seed(args_cli.seed + 1000 * i)
                obs, _ = env.reset()
                shim = DeployShim(policy, **params)
                shim.reset()
                for _ in range(T):
                    a = shim(obs)
                    obs_buf.append(shim.debug_obs_use.cpu())
                    act_buf.append(shim.debug_raw_action.cpu())
                    obs, _, _, _ = env.step(a)
                print(f"[collect] aug {name} done", flush=True)
    else:
        obs, _ = env.reset()
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
