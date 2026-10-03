# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""自定义课程学习（训练期自动升难度/退火；部署期无效）。

设计（[v40 自动课程] 用户要求：把此前的"人工课程"写进代码，一次从头跑到尾）：
  - 信号：env.reward_manager._step_reward（每步已算好的加权项值）里 `lift_success`
    项 ÷ 权重 → 成功率(0~1)，再 EMA（alpha≈0.001，τ≈1000 次调用）。
  - 调用时机：CurriculumManager 在每次环境 reset 时调用（env_ids=本步重置的 env 子集），
    ≈ 每训练步一次；本函数内部用计数器每 `check_every` 次做一次判定。
  - 更新"活对象"（已核实 manager 每步从这些对象读取）：
      event : env.event_manager._terms[name].params[key]
      reward: env.reward_manager._term_cfgs[idx].weight
  - 安全：只升不降；阈值+连续确认+每档最小步数；全部参数可在 CurriculumCfg 单点配置。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 成功率信号来源（reward term 名）
_SIGNAL_TERM = "lift_success"


def _get_event_cfg(env: "ManagerBasedRLEnv", term_name: str):
    """取 EventManager 里的 EventTermCfg（按 mode 遍历；事件每次调用都从 cfg.params 读）。"""
    em = env.event_manager
    for mode, names in em._mode_term_names.items():
        if term_name in names:
            return em._mode_term_cfgs[mode][names.index(term_name)]
    raise KeyError(f"event term 不存在: {term_name}")


def _signal_ema(env: "ManagerBasedRLEnv", alpha: float = 0.001) -> float:
    """读 lift_success 项（归一化 0~1）并做 EMA。失败时保持旧值。"""
    rm = env.reward_manager
    try:
        idx = rm._term_names.index(_SIGNAL_TERM)
        w = max(float(rm._term_cfgs[idx].weight), 1e-6)
        val = float((rm._step_reward[:, idx] / w).mean().clamp(0.0, 1.0))
    except Exception:  # noqa: BLE001
        val = 0.0
    if not hasattr(env, "_cur_ema"):
        env._cur_ema = 0.0
        env._cur_ema_dbg = 0
    env._cur_ema = (1.0 - alpha) * env._cur_ema + alpha * val
    return float(env._cur_ema)


def _tick(env: "ManagerBasedRLEnv", every: int) -> bool:
    """按**环境步**计数（common_step_counter，幂等）：每 every 步返回一次 True。

    多个课程项共享同一计数器；同一训练步内的多次调用返回相同结果。
    """
    cnt = int(getattr(env, "common_step_counter", getattr(env, "_cur_cnt", 0)))
    env._cur_cnt = cnt
    return cnt > 0 and cnt % max(1, int(every)) == 0


def curriculum_arm_far(
    env: "ManagerBasedRLEnv",
    env_ids=None,
    ladder: tuple = (0.0, 0.15, 0.3, 0.5, 0.7, 1.0),
    threshold: float = 0.45,
    min_steps_per_stage: int = 5000,
    check_every: int = 240,
    require_k: int = 3,
    ema_alpha: float = 0.001,
    term_name: str = "reset_arm_far",
    key: str = "scale",
    **kwargs,
) -> float:
    """[v40 自动课程 A] 远起 scale 按表现逐档升到最终值（只升不降）。

    升档条件：success EMA ≥ threshold 且 本档已 ≥min_steps_per_stage 且 连续 require_k 次判定通过。
    初始化时自动把 stage 对齐到 cfg 当前的 scale（续训不会从 0.15 回退）。
    """
    ema = _signal_ema(env, ema_alpha)
    ok = _tick(env, check_every)
    cfg = _get_event_cfg(env, term_name)
    if not hasattr(env, "_cur_far_stage"):
        cur = float(cfg.params.get(key, ladder[0]))
        env._cur_far_stage = min(range(len(ladder)), key=lambda i: abs(float(ladder[i]) - cur))
        env._cur_far_ok = 0
        env._cur_aligned_cnt = 0
        print(f"[curriculum] arm_far 初始化: 当前 scale={cur:.3f} → stage={env._cur_far_stage}/{len(ladder)-1}", flush=True)
    env._cur_aligned_cnt = int(getattr(env, "_cur_aligned_cnt", 0)) + 1
    if ok:
        cond = ema >= threshold and env._cur_aligned_cnt >= min_steps_per_stage
        if cond:
            env._cur_far_ok += 1
        else:
            env._cur_far_ok = 0
        if env._cur_far_ok >= require_k and env._cur_far_stage < len(ladder) - 1:
            env._cur_far_stage += 1
            new = float(ladder[env._cur_far_stage])
            cfg.params[key] = new
            env._cur_far_ok = 0
            env._cur_aligned_cnt = 0
            print(f"[curriculum] arm_far scale → {new:.3f}（stage {env._cur_far_stage}/{len(ladder)-1}, ema={ema:.3f}）", flush=True)
    return float(cfg.params.get(key, ladder[0]))


def curriculum_reward_anneal(
    env: "ManagerBasedRLEnv",
    env_ids=None,
    term_name: str = "thumb_face_reach",
    target_weight: float = 0.0,
    threshold: float = 0.5,
    num_steps: int = 6000,
    check_every: int = 240,
    require_k: int = 3,
    gate_min_scale: float | None = None,
    after_term: str | None = None,
    pause_below: float | None = None,
    ema_alpha: float = 0.001,
    **kwargs,
) -> float:
    """[v40 自动课程 B] 达标后把某项奖励权重线性退火到 target_weight。

    触发：success EMA ≥ threshold 且（可选）远起 scale ≥ gate_min_scale 且（可选）
      after_term 指定的退火项已完成（串行退火），连续 require_k 次。
    过程：记录触发时的权重 w0 与计数 t0，之后按 (cnt−t0)/num_steps 线性过渡。
    """
    ema = _signal_ema(env, ema_alpha)
    rm = env.reward_manager
    idx = rm._term_names.index(term_name)
    key = f"_cur_anneal_{term_name}"
    started = getattr(env, key + "_started", False)
    done = getattr(env, key + "_done", False)
    if not done:
        if _tick(env, check_every) and not started:
            far = float(_get_event_cfg(env, "reset_arm_far").params.get("scale", 0.0))
            after_ok = True if after_term is None else bool(getattr(env, f"_cur_anneal_{after_term}_done", False))
            cond = ema >= threshold and (gate_min_scale is None or far >= gate_min_scale - 1e-6) and after_ok
            n_ok = int(getattr(env, key + "_ok", 0)) + (1 if cond else -max(1, int(getattr(env, key + "_ok", 0))))
            setattr(env, key + "_ok", max(0, n_ok) if cond else 0)
            if cond and int(getattr(env, key + "_ok", 0)) >= require_k:
                setattr(env, key + "_started", True)
                setattr(env, key + "_w0", float(rm._term_cfgs[idx].weight))
                setattr(env, key + "_t0", int(env._cur_cnt))
                print(
                    f"[curriculum] {term_name} 退火触发: {float(getattr(env, key+'_w0')):.3f} → {target_weight}（{num_steps} 次调用）",
                    flush=True,
                )
        if getattr(env, key + "_started", False) and not done:
            w0 = float(getattr(env, key + "_w0"))
            t0 = int(getattr(env, key + "_t0"))
            # [v47 保护] 退化即暂停：EMA 掉到 pause_below 以下时冻结进度（不推进退火）
            if pause_below is not None and ema < float(pause_below):
                shift = int(env._cur_cnt) - t0 - int(getattr(env, key + "_held", 0))
                setattr(env, key + "_held", int(getattr(env, key + "_held", 0)) + shift)
                if int(getattr(env, key + "_held", 0)) % 2000 < 20:
                    print(f"[curriculum] {term_name} 退火暂停(ema={ema:.3f}<{pause_below})", flush=True)
            t0 = int(getattr(env, key + "_t0")) + int(getattr(env, key + "_held", 0))
            frac = min(1.0, max(0.0, (int(env._cur_cnt) - t0) / float(max(1, num_steps))))
            rm._term_cfgs[idx].weight = w0 + (target_weight - w0) * frac
            if frac >= 1.0:
                setattr(env, key + "_done", True)
                print(f"[curriculum] {term_name} 退火完成: weight={target_weight}", flush=True)
    return float(rm._term_cfgs[idx].weight)


def curriculum_param_anneal(
    env: "ManagerBasedRLEnv",
    env_ids=None,
    term_name: str = "contact_gate",
    target: str = "reward",
    key: str = "thr",
    target_value: float = 0.15,
    threshold: float = 0.0,
    num_steps: int = 6000,
    check_every: int = 240,
    require_k: int = 3,
    gate_min_scale: float | None = None,
    after_term: str | None = None,
    pause_below: float | None = None,
    ema_alpha: float = 0.001,
    **kwargs,
) -> float:
    """[v44 自动课程 C] 把某项奖励的**参数**（如门控阈值 thr）从当前值线性渐变到 target。

    用于"先松后严"：thr 0.03 → 0.15（轻触先有密集回报，后期要求逐指到位）。
    触发/串行逻辑同 curriculum_reward_anneal。
    """
    ema = _signal_ema(env, ema_alpha)
    if target == "event":
        term = _get_event_cfg(env, term_name)
    else:
        rm = env.reward_manager
        term = rm._term_cfgs[rm._term_names.index(term_name)]
    astate = f"_cur_anneal_{term_name}_{key}"
    done = getattr(env, astate + "_done", False)
    if not done:
        if _tick(env, check_every) and not getattr(env, astate + "_started", False):
            far = float(_get_event_cfg(env, "reset_arm_far").params.get("scale", 0.0))
            after_ok = True if after_term is None else bool(getattr(env, f"_cur_anneal_{after_term}_done", False))
            cond = ema >= threshold and (gate_min_scale is None or far >= gate_min_scale - 1e-6) and after_ok
            ok_cnt = int(getattr(env, astate + "_ok", 0))
            ok_cnt = ok_cnt + 1 if cond else 0
            setattr(env, astate + "_ok", ok_cnt)
            if cond and ok_cnt >= require_k:
                setattr(env, astate + "_started", True)
                setattr(env, astate + "_v0", float(term.params.get(key, target_value)))
                setattr(env, astate + "_t0", int(env._cur_cnt))
                print(
                    f"[curriculum] {term_name}.{key} 渐变触发: {float(getattr(env, astate+'_v0')):.3f} → {target_value}（{num_steps} 次调用）",
                    flush=True,
                )
        if getattr(env, astate + "_started", False) and not done:
            v0 = float(getattr(env, astate + "_v0"))
            t0 = int(getattr(env, astate + "_t0"))
            if pause_below is not None and ema < float(pause_below):
                shift = int(env._cur_cnt) - t0 - int(getattr(env, astate + "_held", 0))
                setattr(env, astate + "_held", int(getattr(env, astate + "_held", 0)) + shift)
            t0 = t0 + int(getattr(env, astate + "_held", 0))
            frac = min(1.0, max(0.0, (int(env._cur_cnt) - t0) / float(max(1, num_steps))))
            term.params[key] = v0 + (target_value - v0) * frac
            if frac >= 1.0:
                setattr(env, astate + "_done", True)
                setattr(env, f"_cur_anneal_{term_name}_done", True)  # 供 after_term 串行引用
                print(f"[curriculum] {term_name}.{key} 渐变完成: {target_value}", flush=True)
    return float(term.params.get(key, target_value))


def curriculum_unfreeze(
    env: "ManagerBasedRLEnv",
    env_ids=None,
    threshold: float = 0.6,
    num_steps: int = 30000,
    check_every: int = 240,
    require_k: int = 3,
    gate_min_scale: float | None = 1.0,
    after_term: str | None = None,
    pause_below: float | None = None,
    ema_alpha: float = 0.001,
    **kwargs,
) -> float:
    """[v47 软解冻] 达标后把冻结动作通道的放行系数 thaw 0→1 渐变（避免突然放开跳变）。

    历史教训：v45j 二值解冻瞬间策略崩（冻结通道输出从未被约束，放开即大跳）。
    渐变期同样带"退化暂停"（EMA<pause_below 时冻结进度）。完成后 frozen_channels 清空。
    """
    ema = _signal_ema(env, ema_alpha)
    astate = "_cur_anneal_t2_unfreeze"
    am = env.action_manager
    done = getattr(env, astate + "_done", False)
    if not done:
        if _tick(env, check_every) and not getattr(env, astate + "_started", False):
            far = float(_get_event_cfg(env, "reset_arm_far").params.get("scale", 0.0))
            after_ok = True if after_term is None else bool(getattr(env, f"_cur_anneal_{after_term}_done", False))
            cond = ema >= threshold and (gate_min_scale is None or far >= gate_min_scale - 1e-6) and after_ok
            ok_cnt = int(getattr(env, astate + "_ok", 0))
            ok_cnt = ok_cnt + 1 if cond else 0
            setattr(env, astate + "_ok", ok_cnt)
            if cond and ok_cnt >= require_k:
                setattr(env, astate + "_started", True)
                setattr(env, astate + "_t0", int(env._cur_cnt))
                setattr(env, astate + "_held", 0)
                print(f"[curriculum] 软解冻开始: thaw 0→1（{num_steps} 次调用）", flush=True)
        if getattr(env, astate + "_started", False) and not done:
            t0 = int(getattr(env, astate + "_t0"))
            if pause_below is not None and ema < float(pause_below):
                shift = int(env._cur_cnt) - t0 - int(getattr(env, astate + "_held", 0))
                setattr(env, astate + "_held", int(getattr(env, astate + "_held", 0)) + shift)
            t0 = t0 + int(getattr(env, astate + "_held", 0))
            frac = min(1.0, max(0.0, (int(env._cur_cnt) - t0) / float(max(1, num_steps))))
            n = 0
            for name in am.active_terms:
                t = am.get_term(name)
                if hasattr(t.cfg, "thaw"):
                    t.cfg.thaw = frac
                    n += 1
            if frac >= 1.0:
                for name in am.active_terms:
                    t = am.get_term(name)
                    if hasattr(t.cfg, "frozen_channels"):
                        t.cfg.frozen_channels = ()
                setattr(env, astate + "_done", True)
                print(f"[curriculum] 软解冻完成: thaw=1（{n} 个 term）", flush=True)
    return 0.0

def curriculum_gate_lift_anneal(
    env: "ManagerBasedRLEnv",
    env_ids=None,
    target_deadzone: float = 0.35,
    target_scale: float = 0.2,
    threshold: float = 0.5,
    num_steps: int = 6000,
    check_every: int = 240,
    require_k: int = 3,
    gate_min_scale: float | None = None,
    after_term: str | None = None,
    ema_alpha: float = 0.001,
    **kwargs,
) -> float:
    """[v46 自动课程 E] 提起门控参数（GATE_LIFT）线性收紧到最终语义（0.35/0.2）。

    触发：success EMA ≥ threshold（默认 0.5，抓稳后再收紧），支持 after_term 串行。
    """
    from .rewards import GATE_LIFT  # 延迟导入避免环

    ema = _signal_ema(env, ema_alpha)
    astate = "_cur_anneal_gate_lift"
    done = getattr(env, astate + "_done", False)
    if not done:
        if _tick(env, check_every) and not getattr(env, astate + "_started", False):
            far = float(_get_event_cfg(env, "reset_arm_far").params.get("scale", 0.0))
            after_ok = True if after_term is None else bool(getattr(env, f"_cur_anneal_{after_term}_done", False))
            cond = ema >= threshold and (gate_min_scale is None or far >= gate_min_scale - 1e-6) and after_ok
            ok_cnt = int(getattr(env, astate + "_ok", 0))
            ok_cnt = ok_cnt + 1 if cond else 0
            setattr(env, astate + "_ok", ok_cnt)
            if cond and ok_cnt >= require_k:
                setattr(env, astate + "_started", True)
                setattr(env, astate + "_v0", (float(GATE_LIFT["deadzone"]), float(GATE_LIFT["scale"])))
                setattr(env, astate + "_t0", int(env._cur_cnt))
                print(
                    f"[curriculum] GATE_LIFT 收紧触发: {getattr(env, astate+'_v0')} → ({target_deadzone}, {target_scale})（{num_steps} 次调用）",
                    flush=True,
                )
        if getattr(env, astate + "_started", False) and not done:
            v0 = getattr(env, astate + "_v0")
            t0 = int(getattr(env, astate + "_t0"))
            frac = min(1.0, max(0.0, (int(env._cur_cnt) - t0) / float(max(1, num_steps))))
            GATE_LIFT["deadzone"] = v0[0] + (target_deadzone - v0[0]) * frac
            GATE_LIFT["scale"] = v0[1] + (target_scale - v0[1]) * frac
            if frac >= 1.0:
                setattr(env, astate + "_done", True)
                print(f"[curriculum] GATE_LIFT 收紧完成: deadzone={target_deadzone} scale={target_scale}", flush=True)
    return GATE_LIFT["deadzone"]


def curriculum_hand_preclose_anneal(
    env: "ManagerBasedRLEnv",
    env_ids=None,
    offsets: tuple = (
        ("thumb1_joint", -0.249), ("thumb3_joint", 0.420), ("thumb4_joint", -0.633),
        ("index1_joint", -0.080), ("index2_joint", 0.276), ("index3_joint", 0.107),
        ("middle1_joint", -0.080), ("middle2_joint", 0.276), ("middle3_joint", 0.107),
        ("ring1_joint", -0.080), ("ring2_joint", 0.276), ("ring3_joint", 0.107),
        ("little1_joint", -0.080), ("little2_joint", 0.276), ("little3_joint", 0.107),
    ),
    threshold: float = 0.5,
    num_steps: int = 6000,
    check_every: int = 240,
    require_k: int = 3,
    gate_min_scale: float | None = None,
    after_term: str | None = None,
    pause_below: float | None = None,
    ema_alpha: float = 0.001,
    **kwargs,
) -> float:
    """[v46 自动课程 F] 手部"预闭合"：把默认关节位（=动作目标基准）整段前移，让手指
    从"接触前姿态"开始并**保持**（PD 目标一致），解决仅 reset 偏移会被目标拉回的问题。

    首次调用即生效（orig + offsets）；学会抓取（success EMA ≥ threshold）后线性退回 orig。
    与 reset_arm_prepose（臂/腕）配套：两者共同构成"接触预备位"起点，随后全部自动退掉。
    """
    ema = _signal_ema(env, ema_alpha)
    robot = env.scene["robot"]
    astate = "_cur_hand_preclose"
    if not hasattr(env, astate + "_orig"):
        d = robot.data.default_joint_pos
        orig = d.clone()
        ids, offs = [], []
        for name, delta in offsets:
            if name in robot.joint_names:
                i = robot.joint_names.index(name)
                ids.append(i)
                offs.append(float(delta))
                d[:, i] = orig[:, i] + float(delta)
        offs_t = torch.tensor(offs, device=d.device)
        ok = bool(torch.allclose(d[:, ids], orig[:, ids] + offs_t, atol=1e-6))
        setattr(env, astate + "_orig", orig)
        setattr(env, astate + "_ids", ids)
        setattr(env, astate + "_offs", offs_t)
        print(f"[curriculum] 手部预闭合生效: {len(ids)} 个关节 +offsets（覆盖校验={'OK' if ok else '失败!'}）", flush=True)
    done = getattr(env, astate + "_done", False)
    if not done:
        if _tick(env, check_every) and not getattr(env, astate + "_started", False):
            far = float(_get_event_cfg(env, "reset_arm_far").params.get("scale", 0.0))
            after_ok = True if after_term is None else bool(getattr(env, f"_cur_anneal_{after_term}_done", False))
            cond = ema >= threshold and (gate_min_scale is None or far >= gate_min_scale - 1e-6) and after_ok
            ok_cnt = int(getattr(env, astate + "_ok", 0))
            ok_cnt = ok_cnt + 1 if cond else 0
            setattr(env, astate + "_ok", ok_cnt)
            if cond and ok_cnt >= require_k:
                setattr(env, astate + "_started", True)
                setattr(env, astate + "_t0", int(env._cur_cnt))
                print(f"[curriculum] 手部预闭合开始退回（{num_steps} 次调用）", flush=True)
        if getattr(env, astate + "_started", False) and not done:
            orig = getattr(env, astate + "_orig")
            ids = getattr(env, astate + "_ids")
            t0 = int(getattr(env, astate + "_t0"))
            if pause_below is not None and ema < float(pause_below):
                shift = int(env._cur_cnt) - t0 - int(getattr(env, astate + "_held", 0))
                setattr(env, astate + "_held", int(getattr(env, astate + "_held", 0)) + shift)
            t0 = t0 + int(getattr(env, astate + "_held", 0))
            frac = min(1.0, max(0.0, (int(env._cur_cnt) - t0) / float(max(1, num_steps))))
            d = robot.data.default_joint_pos
            offs = getattr(env, astate + "_offs")
            d[:, ids] = orig[:, ids] + (1.0 - frac) * offs
            if frac >= 1.0:
                d[:, ids] = orig[:, ids]
                setattr(env, astate + "_done", True)
                print("[curriculum] 手部预闭合退回完成（默认位=原始预设）", flush=True)
    return 0.0
