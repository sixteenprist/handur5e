# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""σ 护栏（"熵地板"）——项目内 ActorCritic 子类（rsl_rl 3.x）。

背景与目标（[2026-09-17]）：
    训练经典死法之一 = entropy 单调下滑（σ 逐维收缩）→ 新技能所需动作几乎不再被
    采样 → "学不动 / 拉不回来"。entropy_coef 只是"推力"（把 σ 平衡点往大挪，
    σ*≈sqrt(coef/C)），当回报曲率 C 大（悬停对噪声敏感、action_rate∝σ²、空闲维只有
    成本没有收入）时推不过去——实证：coef 0.0015→0.005，熵仍塌到 -13（σ≈0.10）。
    本类给"约束"：把 σ 夹在 [floor, cap]，跌出下界即被拉回 —— 保证任何时刻都保留
    最小探索预算（"不让训练过早坍缩、随时能回来"）。

    （数学说明：对角高斯 H_i = ln σ_i + 1.419，熵对 σ 单调 —— "限制熵下限"在实现上
    就是"限制 σ 下限"；区别只在数值怎么定、分不分维度、硬还是软，不在机制。
    软地板（低于阈值才加大推力）需要子类 PPO 的 loss，属后续可选项。）

设计要点：
    - 分组：arm=[0:6] / hand=[6:14]（与 scripts/rsl_rl/sigma_surgery.py 同口径）。
      历史病是"手部维（尤其拇指）先塌"；臂部是悬停精度的命门——不该全局加噪。
    - 参数级 in-place 夹取（而非采样处临时夹）：σ 停在边界时仍能被梯度再抬升
      （"活地板"）；且 TB 的 noise_std / entropy 读数与真实采样一致。
    - 界值用非持久 buffer 存放：不进 state_dict → 与所有旧 checkpoint 兼容
      （resume 不报 missing/unexpected keys）。
    - 默认全部关闭（-1）：行为与原生 ActorCritic 完全一致（先冒烟、后启用）。
    - 兼容 noise_std_type = "scalar"（本项目：σ 直接是参数）与 "log"。

启用方式（改下面常量，单变量纪律）：
    1) 当前采用：STD_FLOOR_ARM = STD_FLOOR_HAND = 0.12 —— 总熵（14 维求和 Σ(lnσ+1.419)）全贴地板
       ≈ -9.8，即"熵不会再跌到 -10 以下"（用户定额：-12 太低、-10 差不多）。
       换算参考：σ=0.10 ↔ 总熵 -12.4；σ=0.074 ↔ -16.5（枯竭线）。
       换算参考：σ=0.10 ↔ 总熵 -12.4；σ=0.074 ↔ -16.5（枯竭线）。
    2) 防"熵爆家族"（历史 0.42→0.74）的 STD_CAP_*（如 0.5，当前不触动）——先留空，需要再开。

不动 site-packages：rsl_rl 的 runner 用 eval(class_name) 在自身模块命名空间找策略类
（v3.0.1 源码核实），本模块提供 register_clamped_actor_critic() 由项目侧（rsl_rl_ppo_cfg.py
导入时）把类注册进去。不用时把 class_name 改回 "ActorCritic" 即可。
"""

from __future__ import annotations

import torch

from rsl_rl.modules import ActorCritic

# ===== 启用开关：σ 界值（单位 = σ 本身；-1.0 = 该组不启用）=====
# [2026-09-17 晚·定稿] 两组地板均设 0.12（目标：总熵 ≈ -10，换算见 docstring）。
#   臂部同守的原因：总熵是 14 维求和——只守手部时，臂维继续缩会把总值再拖低。
# [2026-09-19 分组修正·依据逐维 σ 实证] 一刀切 0.12/0.25 的两个问题：
#   ① 臂部 σ 被推到 0.22（×scale 0.005 = 1.1mm/步噪声）——悬停精度被噪声毁，
#      value 尖峰常在其后；臂只需精度、不需探索 → 压死：0.08/0.13。
#   ② 手部 σ 被 entropy 从 0.14 推到 0.20+，接触学习期（finger_contact 0.02 时）
#      叠加最大噪声 → 一次 update 打穿（2026-09-19 run：764 → 766 暴毙）。
#      → 手部给探索空间但设上限：0.14/0.20（仍高于旧收敛点 0.14）。
# [2026-09-20 v4 低引导·用户决定] 降噪：arm 0.09 / hand 0.12（原 0.10/0.16）。
#   两指捏取（拇指+中指）动作空间小、目标明确，不需要 9° 的探索噪声；
#   play 实测 0.16 导致"手指抖动不稳定"。0.12≈7° 起步，仍保留探索。
#   仍为固定值（floor=cap）：本项目 σ 自由学习从未稳定（膨胀/触底反弹均崩）。
# [2026-09-21 v5.8 用户反馈修正] 回到 0.09/0.12（v4/v5 基线，用户原设置）：
#   v5.4 曾自主降到 0.06/0.08（"保护均值策略"）——用户指出这是把 σ 当主要矛盾、
#   越调越偏。真正的主因已修复：① lr 改动此前被优化器状态覆盖（train.py 已打补丁）；
#   ② rollout 16→32 改善长程信用分配。σ 恢复健康探索预算（0.09≈5.2° / 0.12≈6.9°），
#   稳定性交给真修的两条。回退/微调：0.06/0.08（保守）或 0.12/0.16（更探索）。
STD_FLOOR_ARM: float = 0.09
STD_FLOOR_HAND: float = 0.12
STD_CAP_ARM: float = 0.09
STD_CAP_HAND: float = 0.12

# 动作维度分组（本项目 14 维 = arm 6 + hand 8；与 sigma_surgery.py 同口径）
_ARM_SLICE = (0, 6)


def _build_bounds(n_actions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """按组生成逐维 [lo, hi] 界（禁用组用"数值无效界"占位）。"""
    lo = torch.zeros(n_actions)
    hi = torch.full((n_actions,), 1e6)

    a0, a1 = _ARM_SLICE
    if STD_FLOOR_ARM >= 0:
        lo[a0:a1] = STD_FLOOR_ARM
    if STD_CAP_ARM >= 0:
        hi[a0:a1] = STD_CAP_ARM
    if STD_FLOOR_HAND >= 0:
        lo[a1:] = STD_FLOOR_HAND
    if STD_CAP_HAND >= 0:
        hi[a1:] = STD_CAP_HAND
    return lo, hi


class ClampedActorCritic(ActorCritic):
    """ActorCritic + σ 分组夹取（其余行为与原生完全一致）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n = int(self.std.numel()) if self.noise_std_type == "scalar" else int(self.log_std.numel())
        lo, hi = _build_bounds(n)
        # 非持久 buffer：随 .to(device) 迁移、不进 state_dict（旧 ckpt 直接兼容）
        self.register_buffer("_std_lo", lo, persistent=False)
        self.register_buffer("_std_hi", hi, persistent=False)

        if max(STD_FLOOR_ARM, STD_FLOOR_HAND, STD_CAP_ARM, STD_CAP_HAND) >= 0:
            print(
                "[ClampedActorCritic] σ 护栏启用："
                f"floor(arm)={STD_FLOOR_ARM} floor(hand)={STD_FLOOR_HAND} "
                f"cap(arm)={STD_CAP_ARM} cap(hand)={STD_CAP_HAND}（-1=关）"
            )
        else:
            print("[ClampedActorCritic] σ 护栏未启用（全 -1，行为 = 原生 ActorCritic）")

    def update_distribution(self, obs):
        # 参数级夹取：越界立即拉回；停在边界时梯度仍可把它再抬升 → "活地板"
        with torch.no_grad():
            if self.noise_std_type == "scalar":
                self.std.clamp_(min=self._std_lo, max=self._std_hi)
            else:  # "log"
                self.log_std.clamp_(
                    min=torch.log(self._std_lo.clamp(min=1e-8)),  # 禁用位 ≈ -18.4，永不生效
                    max=torch.log(self._std_hi),
                )
        super().update_distribution(obs)


def register_clamped_actor_critic() -> None:
    """把本类注册进 rsl_rl runner 模块命名空间（eval(class_name) 才能找到）。

    rsl_rl v3 的 OnPolicyRunner 用 `eval(self.policy_cfg.pop("class_name"))` 实例化策略类，
    自定义类必须在该模块的 globals 里可见——这里从项目侧注入，不触碰 site-packages。
    """
    try:
        import rsl_rl.runners.on_policy_runner as _runner_mod

        _runner_mod.ClampedActorCritic = ClampedActorCritic
    except Exception as exc:  # noqa: BLE001
        # 注册失败不影响 class_name="ActorCritic" 的默认路径；若随后用了自定义类名会明确报 NameError
        print(f"[clamped_actor_critic] 注册失败（不影响默认 ActorCritic 路径）：{exc}")
