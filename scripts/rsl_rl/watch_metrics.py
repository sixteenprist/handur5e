#!/usr/bin/env python3
# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""rsl_rl 训练监控（v3 抓取链）：每 N 秒解析最新 events，输出换算读数 + 告警。

用法（训练机）:
  mkdir -p logs/monitor/ur5e_grasp3_v1
  python -u scripts/rsl_rl/watch_metrics.py \
      --logdir logs/rsl_rl/ur5e_grasp3_v1 \
      --monitordir logs/monitor/ur5e_grasp3_v1 \
      --interval 300  > logs/monitor/ur5e_grasp3_v1/monitor_$(date +%Y%m%d_%H%M%S).log 2>&1 &

读数换算（Episode_Reward/权重 = 时间占比）:
  succ% = lift_success/5   提起达标且抓握占比（>0.5 算成功）
  gate% = contact_gate/2   对向接触占比（>0.7 抓稳）
  lift% = lift_height/15   平均净提升比例（>0.8）
  reach = reach/2          整手靠近（应最早饱和 >0.95）

告警（去重，状态存 <monitordir>/state.json）:
  value loss >1.0 / collection >4.5s / mean_reward 峰值回撤 >30%
  abnormal_robot 或 fingertip_overload 终止触发
  里程碑: gate%>0.2（首次接触）、succ%>0.5（首次提起成功）
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYS = [
    "Train/mean_reward",
    "Loss/value_function",
    "Loss/surrogate",
    "Policy/mean_noise_std",
    "Perf/collection time",
    "Episode_Reward/reach",
    "Episode_Reward/contact_gate",
    "Episode_Reward/lift_height",
    "Episode_Reward/lift_success",
    "Episode_Termination/abnormal_robot",
    "Episode_Termination/fingertip_overload",
]


def fmt(v, nd=3):
    return "  -  " if v is None else f"{v:.{nd}g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="rsl_rl 实验目录（其下按时间戳分 run）")
    ap.add_argument("--monitordir", default=None, help="监控状态/日志目录（默认 logdir 同级）")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--vloss_alert", type=float, default=1.0)
    ap.add_argument("--collect_alert", type=float, default=4.5)
    args = ap.parse_args()

    monitordir = args.monitordir or os.path.join("logs", "monitor", os.path.basename(os.path.normpath(args.logdir)))
    os.makedirs(monitordir, exist_ok=True)
    state_path = os.path.join(monitordir, "state.json")
    try:
        with open(state_path) as f:
            st = json.load(f)
    except Exception:
        st = {}
    st.setdefault("vloss_step", -1)
    st.setdefault("peak_reward", 0.0)
    st.setdefault("drawdown_alerted", False)
    st.setdefault("collect_alerted", False)
    st.setdefault("term_alerted", False)
    st.setdefault("gate_ms", False)
    st.setdefault("succ_ms", False)

    print(f"[watch] 启动 logdir={args.logdir} monitordir={monitordir} interval={args.interval}s", flush=True)

    while True:
        stamp = datetime.now().strftime("%m-%d %H:%M:%S")
        try:
            files = glob.glob(os.path.join(args.logdir, "*", "events.out.tfevents.*"))
            if not files:
                print(f"[{stamp}] 等待 events 出现...", flush=True)
                time.sleep(args.interval)
                continue
            f = max(files, key=os.path.getmtime)
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
            tags = set(ea.Tags()["scalars"])
            data = {k: {e.step: e.value for e in ea.Scalars(k)} for k in KEYS if k in tags}
            vl = data.get("Loss/value_function", {})
            steps = sorted(vl)
            if not steps:
                print(f"[{stamp}] events 为空，等待...", flush=True)
                time.sleep(args.interval)
                continue
            it = steps[-1]
            win = [vl[s] for s in steps if s > it - 300]
            vmax = max(win) if win else 0.0

            def latest(tag):
                d = data.get(tag, {})
                return d[max(d)] if d else None

            rew = latest("Train/mean_reward")
            reach = latest("Episode_Reward/reach")
            gate = latest("Episode_Reward/contact_gate")
            lift_h = latest("Episode_Reward/lift_height")
            lift_s = latest("Episode_Reward/lift_success")
            collect = latest("Perf/collection time")
            surr = latest("Loss/surrogate")
            sig = latest("Policy/mean_noise_std")

            line = (
                f"[{stamp}] iter={it:5d} | rew={fmt(rew, 4)} | vloss={fmt(vl.get(it))} (300max={fmt(vmax)}) | "
                f"surr={fmt(surr)} | collect={fmt(collect, 2)}s | sig={fmt(sig)} | "
                f"reach/2={fmt(None if reach is None else reach / 2)} | gate/2={fmt(None if gate is None else gate / 2)} | "
                f"liftH/15={fmt(None if lift_h is None else lift_h / 15)} | liftS/5={fmt(None if lift_s is None else lift_s / 5)}"
            )
            print(line, flush=True)

            alerts = []
            if vl.get(it, 0.0) > args.vloss_alert and it > st["vloss_step"]:
                alerts.append(f"⚠️ value_loss {vl[it]:.2f} @ iter {it}（阈值 {args.vloss_alert}）")
                st["vloss_step"] = it
            if rew is not None:
                st["peak_reward"] = max(st["peak_reward"], rew)
                if st["peak_reward"] > 5 and rew < 0.7 * st["peak_reward"] and not st["drawdown_alerted"]:
                    alerts.append(f"⚠️ mean_reward 从峰值 {st['peak_reward']:.1f} 回撤到 {rew:.1f}")
                    st["drawdown_alerted"] = True
            if collect is not None and collect > args.collect_alert and not st["collect_alerted"]:
                alerts.append(f"⚠️ collection {collect:.2f}s > {args.collect_alert}（物理挣扎）")
                st["collect_alerted"] = True
            term = (latest("Episode_Termination/abnormal_robot") or 0) + (latest("Episode_Termination/fingertip_overload") or 0)
            if term > 0.0 and not st["term_alerted"]:
                alerts.append(f"⚠️ 异常终止触发（abnormal/overload 合计 {term:.3f}/episode）")
                st["term_alerted"] = True
            if gate is not None and gate / 2 > 0.2 and not st["gate_ms"]:
                alerts.append(f"ℹ️ 里程碑：contact_gate>0.2（= {gate/2:.2f}）—— 首次稳定对向接触")
                st["gate_ms"] = True
            if lift_s is not None and lift_s / 5 > 0.5 and not st["succ_ms"]:
                alerts.append(f"ℹ️ 里程碑：lift_success>0.5（= {lift_s/5:.2f}）—— 提起达标")
                st["succ_ms"] = True
            for a in alerts:
                print(f"[{stamp}] {a}", flush=True)

            with open(state_path, "w") as fh:
                json.dump(st, fh)
        except Exception as exc:  # noqa: BLE001
            print(f"[{stamp}] 读取异常（跳过本轮）: {exc}", flush=True)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
