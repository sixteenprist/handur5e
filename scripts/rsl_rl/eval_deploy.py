# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""部署条件评测 harness：在策略与仿真之间注入"真机不完美"，量化部署裕度。

注入项（单项或组合，均为策略侧"垫片"，不动环境/不动权重）：
  - 动作低通滤波 : a' = β·a + (1−β)·a_prev        （真机伺服滞后/柔化）
  - 手臂步间限幅 : arm raw 每步变化 ≤ rate          （真机速度/加速度限制）
  - 观测延迟     : obs 晚 D 步送达                  （通信/感知延迟）
  - 观测噪声     : 关节位置/速度加高斯噪声          （编码器噪声）

附加诊断：
  - 保持期手颤   : 已提起后手部 20 关节速度 RMS（rad/s）。对比 baseline 可判断
                   低通滤波能否压制颤动；若压不动 → 颤动来自接触/物理而非动作抖动。

用途
----
1. 真机联调前在仿真内暴露掉分点（哪个注入最致命 → 决定 DR/桥的优先级）；
2. 验证"部署侧低通滤波"能否顺带治疗手指颤动。

用法（项目根目录）
------------------
    # 全套矩阵（baseline + 16 条件 = 17 场景；含动作延迟）：
    python scripts/rsl_rl/eval_deploy.py \
        --checkpoint logs/rsl_rl/ur5e_grasp_v5/best/grasp_v58_delayrobust.pt \
        --num_envs 100 --steps 300 --scenario all

    # 单项自定义：
    python scripts/rsl_rl/eval_deploy.py --checkpoint ... --scenario custom \
        --beta 0.5 --delay 2 --rate-arm 0.5 --noise-pos 0.002 --noise-vel 0.01

实现说明：环境与策略**只建一次**，场景之间用 ``env.reset()`` 切换（每场景重建
USD stage 会累积内存且实测卡死）。评估口径与 ``eval_final.py`` 一致
（成功=提起净高度≥8cm，确定性策略，最终部署配置）。
"""

import argparse
import collections

parser = argparse.ArgumentParser(description="部署条件评测 harness（策略侧注入真机不完美）")
parser.add_argument("--task", type=str, default="Template-Ur5e-Drillgrasp-v0", help="任务名")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/rsl_rl/ur5e_grasp_v5/best/grasp_v58_delayrobust.pt",
    help="权重文件路径（.pt）",
)
parser.add_argument("--num_envs", type=int, default=100, help="评测环境数 = 集数（默认 100）")
parser.add_argument("--steps", type=int, default=300, help="每集步数（默认 300 ≈ 10s@30Hz）")
parser.add_argument("--scenario", type=str, default="all", help="all / baseline / 单项名 / custom")
parser.add_argument("--beta", type=float, default=None, help="动作低通系数（1.0=不过滤；越小越平滑）")
parser.add_argument("--delay", type=int, default=0, help="观测延迟步数（0=无延迟）")
parser.add_argument("--act-delay", type=int, default=0, help="动作延迟步数（命令晚 N 步生效；0=无）")
parser.add_argument("--rate-arm", type=float, default=None, help="手臂 raw 步间变化上限（None=不限制）")
parser.add_argument("--noise-pos", type=float, default=0.0, help="关节位置观测噪声 σ（rad）")
parser.add_argument("--noise-vel", type=float, default=0.0, help="关节速度观测噪声 σ（rad/s）")
parser.add_argument("--seed", type=int, default=42, help="每个场景 reset 前的随机种子（保证各场景起点可比）")
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

SUCCESS_HEIGHT = 0.08  # 与 eval_final.py / lift_success_reward 对齐
N_POS, N_VEL = 26, 26  # obs 前 52 维 = 关节位置差 + 关节速度（见 DEPLOY_INTERFACE.md §2）

# 默认场景矩阵：单项 + 组合（组合 = 中档三件套）
# [2026-10-14 第一轮实测后的补充] rate 0.5 全崩 → 补 1.5/2.0 找良性阈值；
#   补 filter+delay1 现实组合；filter_b0.7 看更温和的滤波是否零损。
SCENARIOS = {
    "baseline": {},
    "filter_b0.7": {"beta": 0.7},
    "filter_b0.5": {"beta": 0.5},
    "filter_b0.33": {"beta": 0.33},
    "delay_d1": {"delay": 1},
    "delay_d2": {"delay": 2},
    "delay_d4": {"delay": 4},
    "noise": {"noise_pos": 0.002, "noise_vel": 0.01},
    "rate_arm0.5": {"rate_arm": 0.5},
    "rate_arm1.5": {"rate_arm": 1.5},
    "rate_arm2.0": {"rate_arm": 2.0},
    "filter_b0.5_d1": {"beta": 0.5, "delay": 1},
    "combo_b0.5_d2_noise": {"beta": 0.5, "delay": 2, "noise_pos": 0.002, "noise_vel": 0.01},
    # [2026-10-15 补测] 动作延迟（命令晚 N 步才生效；真机传输+执行器滞后的另一半）
    "act_d1": {"act_delay": 1},
    "act_d2": {"act_delay": 2},
    "act_d1_b0.5": {"act_delay": 1, "beta": 0.5},
    "act_d1_obs_d1": {"act_delay": 1, "delay": 1},
}


class DeployShim:
    """策略垫片：观测延迟/加噪 → 策略 → 动作滤波/限幅。"""

    def __init__(
        self,
        policy,
        beta=None,
        delay=0,
        rate_arm=None,
        noise_pos=0.0,
        noise_vel=0.0,
        act_delay=0,
    ):
        self.policy = policy
        self.beta = beta
        self.delay = int(delay)
        self.rate_arm = rate_arm
        self.act_delay = int(act_delay)
        self._act_buf = []
        self.noise_pos = float(noise_pos)
        self.noise_vel = float(noise_vel)
        self._hist = None
        self._prev = None

    def reset(self):
        """episode 开始时清空延迟缓冲与滤波状态（模拟真机重启后无历史）。"""
        self._hist = collections.deque(maxlen=self.delay + 1) if self.delay > 0 else None
        self._act_buf = []
        self._prev = None

    def __call__(self, obs):
        # obs 是 TensorDict（键 'policy'/'critic'，见 probe_obs_type）：垫片只操作 policy 张量
        t = obs["policy"]

        # 1) 观测延迟：延迟未满时用最旧样本（warmup 行为，真机等同"刚上电还没新帧"）
        if self._hist is not None:
            self._hist.append(t.detach().clone())
            obs_use = self._hist[0]
        else:
            obs_use = t

        # 2) 观测噪声：只加在前 52 维（关节位置差/速度）
        if self.noise_pos > 0.0 or self.noise_vel > 0.0:
            obs_use = obs_use.clone()
            if self.noise_pos > 0.0:
                obs_use[:, 0:N_POS] += torch.randn_like(obs_use[:, 0:N_POS]) * self.noise_pos
            if self.noise_vel > 0.0:
                obs_use[:, N_POS : N_POS + N_VEL] += (
                    torch.randn_like(obs_use[:, N_POS : N_POS + N_VEL]) * self.noise_vel
                )

        if self._hist is not None or self.noise_pos > 0.0 or self.noise_vel > 0.0:
            obs_td = obs.clone()
            obs_td["policy"] = obs_use
            a = self.policy(obs_td)
        else:
            a = self.policy(obs)

        if self._prev is None:
            self._prev = a.detach().clone()

        # 3) 动作低通：EMA（用上一发出值，含已被限幅的结果 → 与真机伺服一致）
        if self.beta is not None and self.beta < 1.0:
            a = self.beta * a + (1.0 - self.beta) * self._prev

        # 4) 手臂步间限幅（raw 空间；手部已有环境内 max_delta=0.3 保护，不重复施加）
        if self.rate_arm is not None:
            a = a.clone()
            d = (a[:, :6] - self._prev[:, :6]).clamp(-self.rate_arm, self.rate_arm)
            a[:, :6] = self._prev[:, :6] + d

        # 5) 动作延迟：命令晚 act_delay 步才生效（真机链路顺序：滤波→限幅→传输/执行器滞后）
        if self.act_delay > 0:
            self._act_buf.append(a.detach().clone())
            if len(self._act_buf) <= self.act_delay:
                a = torch.zeros_like(a)          # warmup=空命令（与训练 wrapper 的重置填充一致）
            else:
                a = self._act_buf.pop(0)

        self._prev = a.detach().clone()
        return a


def run_scenario(env, cube, policy, name: str, params: dict, N: int, T: int) -> dict:
    """单场景：reset → 施加注入跑 N 集 → 返回指标（环境/策略复用，避免重建 stage）。"""
    shim = DeployShim(policy, **params)

    torch.manual_seed(args_cli.seed)  # 各场景起点对齐
    obs, _ = env.reset()
    shim.reset()
    cp0 = cube.data.root_pos_w.clone()
    device = env.unwrapped.device

    zmax = torch.zeros(N, device=device)
    dmax = torch.zeros(N, device=device)
    d_at = torch.full((N,), float("nan"), device=device)
    f_at = torch.zeros(N, device=device)
    f4_at = torch.zeros(N, device=device)
    trem_sum = torch.zeros(N, device=device)  # 保持期(已提起)手部关节速度 RMS 累积
    trem_cnt = torch.zeros(N, device=device)

    with torch.no_grad():
        for _ in range(T):
            a = shim(obs)
            obs, _, _, _ = env.step(a)
            fv = torch.norm(R._fingertip_force_vectors(env.unwrapped), dim=-1)
            dz = cube.data.root_pos_w[:, 2] - cp0[:, 2]
            dxy = torch.norm((cube.data.root_pos_w - cp0)[:, :2], dim=-1)
            newly = (zmax < SUCCESS_HEIGHT) & (dz >= SUCCESS_HEIGHT)
            if newly.any():
                d_at[newly] = dxy[newly]
                f_at[newly] = fv[newly, 0]
                f4_at[newly] = fv[newly, 1:].max(dim=1).values
            # 手颤指标：已提起的步里，手部 20 关节速度 RMS（rad/s）。↓ = 更稳
            hold = zmax >= SUCCESS_HEIGHT
            if hold.any():
                vh = env.unwrapped.scene["robot"].data.joint_vel[:, 6:]
                rms = torch.sqrt((vh * vh).mean(dim=1))
                trem_sum[hold] += rms[hold]
                trem_cnt[hold] += 1
            zmax = torch.maximum(zmax, dz)
            dmax = torch.maximum(dmax, dxy)

    lift = zmax * 100
    dr = dmax * 100
    succ = zmax >= SUCCESS_HEIGHT
    tremor = torch.where(trem_cnt > 0, trem_sum / trem_cnt.clamp(min=1.0), torch.zeros_like(trem_sum))

    return {
        "scenario": name,
        "succ": float(succ.float().mean()) * 100,
        "n_succ": int(succ.sum()),
        "lift_mean": float(lift[succ].mean()) if succ.any() else 0.0,
        "lift_min": float(lift[succ].min()) if succ.any() else 0.0,
        "drift_mean": float(dr.mean()),
        "drift_max": float(dr.max()),
        "f_thumb": float(f_at[succ].mean()) if succ.any() else 0.0,
        "f_four": float(f4_at[succ].mean()) if succ.any() else 0.0,
        "tremor": float(tremor[succ].mean()) if succ.any() else 0.0,
    }


def main():
    if args_cli.scenario == "all":
        todo = list(SCENARIOS.items())
    elif args_cli.scenario in SCENARIOS:
        todo = [(args_cli.scenario, SCENARIOS[args_cli.scenario])]
    elif args_cli.scenario == "custom":
        todo = [
            (
                "custom",
                {
                    "beta": args_cli.beta,
                    "delay": args_cli.delay,
                    "rate_arm": args_cli.rate_arm,
                    "noise_pos": args_cli.noise_pos,
                    "noise_vel": args_cli.noise_vel,
                    "act_delay": args_cli.act_delay,
                },
            )
        ]
    else:
        raise ValueError(f"未知场景 {args_cli.scenario!r}；可选: all / {' / '.join(SCENARIOS)} / custom")

    N, T = args_cli.num_envs, args_cli.steps
    print("=" * 72, flush=True)
    print(f"部署条件评测（N={N}，每集 {T} 步，确定性策略，最终部署配置）", flush=True)
    print(f"  权重: {args_cli.checkpoint}", flush=True)
    print("=" * 72, flush=True)

    # 环境与策略只建一次（见文件头"实现说明"）
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=N)
    apply_final_deploy_overrides(cfg)
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=cfg))
    runner = OnPolicyRunner(env, PPORunnerCfg().to_dict(), log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    cube = env.unwrapped.scene["cube_obj"]

    results = []
    for name, params in todo:
        p_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "无注入"
        print(f"\n── 场景 [{name}] ({p_str}) 运行中…", flush=True)
        r = run_scenario(env, cube, policy, name, params, N, T)
        results.append(r)
        print(
            f"   成功率 {r['n_succ']}/{N} = {r['succ']:.1f}% | "
            f"提起 mean {r['lift_mean']:.1f} / min {r['lift_min']:.1f} cm | "
            f"漂移 mean {r['drift_mean']:.1f} / max {r['drift_max']:.1f} cm | "
            f"抓取力 拇 {r['f_thumb']:.2f} / 四max {r['f_four']:.2f} N | "
            f"保持期手颤 {r['tremor']:.3f} rad/s",
            flush=True,
        )

    print("\n" + "=" * 72, flush=True)
    print("汇总表（真机部署裕度）", flush=True)
    print(
        f"{'场景':<22}{'成功率':>8}{'提起mean':>11}{'提起min':>10}{'漂移mean':>11}{'漂移max':>10}{'手颤':>7}",
        flush=True,
    )
    for r in results:
        print(
            f"{r['scenario']:<22}{r['succ']:>7.1f}%{r['lift_mean']:>9.1f}cm{r['lift_min']:>8.1f}cm"
            f"{r['drift_mean']:>9.1f}cm{r['drift_max']:>8.1f}cm{r['tremor']:>7.3f}",
            flush=True,
        )
    print("=" * 72, flush=True)
    env.close()
    sim.close()


if __name__ == "__main__":
    main()
