# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""硬编程 Probe #1：零动作基线 + 手部 8 通道逐维标定（ur5e_drillgrasp）。

[2026-09-18 新建] 硬编程"接近→抓取→提起"全链的第一段：只借环境的动作/物理/
传感器链路（与 RL 相同的 14D action → OSC/PD → 关节），不加载策略、不训练。

设计原则（与用户讨论对齐）:
  - **一切从「零动作点」开始**：等同训练 episode 第 0 步——不施加任何动作、
    不做预设/接近；臂由 SafeOSC 保持（零动作=追当前位姿），手停在默认位姿。
  - 全程只动"手部 8 通道"（thumb1-4 + 四指共享 1-4），一次一维、±幅对称，
    记录「哪些关节动了 + 指尖往哪走」→ 为全链的"拇指绕-合两段式"、
    "中指接近-接触"时序提供经验方向（不靠命名猜符号、不靠猜方向）。
  - 臂自始至终零动作（不动）；无接触（力应恒 0，>0.05N 会警告）。
  - [2026-09-19 v2] 实测零动作腕漂移 ≈+1.38 rad/79°/100s（wrist_3，速率缓降未停）→ world 系指尖位移
    被其污染（无关节运动的窗口位移仍有 5mm 级）；v2 起位移改以 **wrist_3_link 局部系（handΔ）** 为主输出
    ——整手刚体漂移在局部系自动抵消，只剩纯关节效应（CSV 同时保留 worldΔ 供存档）。
  - [2026-09-19 v3] 增加每通道的**指尖姿态增量 rotΔ**（腕局部系、单位度）：回答"哪个通道/符号在转动
    指腹朝向"（thumb4=指腹转角盘；四指卷曲时指腹怎么倒）——位移+姿态两手数据齐了再定绕-合时序。

用法（远程 conda isaacsim5.1；与 zero_agent.py / debug_tcp_pose.py 相同）:
    python scripts/hand_probe.py --task Template-Ur5e-Drillgrasp-v0 --num_envs 1
    （任务注册名 = gym.register 的 id；旧写法 ur5e_drillgrasp 未注册会 NameNotFound）

输出:
    - 控制台：P0 基线快照（reset 瞬间 + settle 后）+ P1 每通道汇总
      （关节位移 / 五指指尖位移 cm / 指尖姿态增量 deg / 最大接触力）
    - CSV: <out_dir>/hand_probe_baseline_*.csv、hand_probe_channels_*.csv
      （把控制台文本或这两份 CSV 发回即可）
"""

import argparse
import csv
import os
import time

from isaaclab.app import AppLauncher

# ── argparse ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Hand probe: zero-action baseline + per-channel calibration.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Ur5e-Drillgrasp-v0", help="Name of the task.")  # [2026-09-19 修正] 注册名（原 ur5e_drillgrasp 未注册）
parser.add_argument("--settle_steps", type=int, default=60, help="零动作 settle 步数（基线前）。")
parser.add_argument("--hold_steps", type=int, default=20, help="每个通道命令后保持的步数（测量窗口）。")
parser.add_argument("--amp", type=float, default=0.2, help="通道标定幅度（rad，对称 ±）。")
parser.add_argument("--out_dir", type=str, default="hand_probe_out", help="CSV 输出目录。")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply_inverse  # [v2] 指尖位移腕局部系换算

import ur5e_drillgrasp.tasks  # noqa: F401
from ur5e_drillgrasp.tasks.manager_based.ur5e_drillgrasp.mdp import observations as mdp_obs


def main():
    """零动作基线 + 8 通道逐维标定。"""
    # ── 环境（照 zero_agent.py / debug_tcp_pose.py 模式）──────────────────
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # probe 全程 ~700 步（≈23s @30Hz）：拉长 episode 防中途重置打断测量
    env_cfg.episode_length_s = 120.0
    env = gym.make(args_cli.task, cfg=env_cfg)
    uw = env.unwrapped
    dev = uw.device

    robot = uw.scene["robot"]
    cube = uw.scene["cube_obj"]

    # ── 解析动作空间布局（从 action_manager 动态读取，防硬编码顺序出错）──
    am = uw.action_manager
    terms = getattr(am, "_terms", None)
    if terms is None:  # 兼容不同版本
        terms = {n: am.get_term(n) for n in am.active_terms}
    layout, off = {}, 0
    for n, t in terms.items():
        layout[n] = (off, t.action_dim)
        off += t.action_dim
    total_dim = off
    print("[PROBE] 动作项布局: " + " | ".join(
        f"{n}@{layout[n][0]}:{layout[n][0] + layout[n][1]}(dim{layout[n][1]})" for n in layout))
    hand_name = next(n for n in layout if "hand" in n.lower())
    hand_off, hand_dim = layout[hand_name]
    hand_term = terms[hand_name]
    n_thumb = int(getattr(hand_term, "_n_thumb", 4))
    thumb_ids = [int(i) for i in getattr(hand_term, "_thumb_ids", [])]
    four_map = [list(g) for g in getattr(hand_term, "_four_map", [])]
    joint_names = list(robot.joint_names)
    # 臂关节索引：零动作下会缓慢重力漂移（2026-09-02 侧翻诊断，无恢复力所致）→ 下面做漂移计量
    arm_joint_idx = [i for i, nm in enumerate(joint_names) if any(k in nm for k in ("shoulder", "elbow", "wrist"))]

    channels = []  # [(label, action_index)]
    for i in range(n_thumb):
        jn = joint_names[thumb_ids[i]] if i < len(thumb_ids) else f"thumb{i + 1}"
        channels.append((f"thumb{i + 1}[{jn}]", hand_off + i))
    for i in range(4):
        jn = "&".join(joint_names[j] for j in four_map[i]) if i < len(four_map) else f"four{i + 1}"
        channels.append((f"four{i + 1}[{jn}]", hand_off + n_thumb + i))
    print(f"[PROBE] hand 项={hand_name} @{hand_off}，dim={hand_dim}（n_thumb={n_thumb}）；"
          f"总动作维度={total_dim}；num_envs={uw.num_envs}")
    for lbl, idx in channels:
        print(f"[PROBE]   通道 idx {idx}: {lbl}")

    # ── 指尖 body id / 状态快照工具 ─────────────────────────────────────
    tip_names = list(mdp_obs._FINGERTIP_NAMES)  # ["thumb4","index4","middle4","ring4","little4"]
    body_id = {}
    for nm in tip_names:
        ids, _ = robot.find_bodies(nm)
        if len(ids) == 0:
            raise RuntimeError(f"找不到指尖 body: {nm}（资产命名变了？）")
        body_id[nm] = int(ids[0])
    wrist_ids, _ = robot.find_bodies("wrist_3_link")
    wrist_id = int(wrist_ids[0]) if len(wrist_ids) else None

    def snapshot():
        """读一帧全状态（指尖世界坐标 + wrist_3_link 局部坐标 / 关节 / 力 / cube / 腕 / TCP）。"""
        with torch.inference_mode():
            tips = {nm: robot.data.body_link_pos_w[0, body_id[nm]].clone() for nm in tip_names}
            tip_q = {nm: robot.data.body_link_quat_w[0, body_id[nm]].clone() for nm in tip_names}  # [v3]
            wrist_p = robot.data.body_link_pos_w[0, wrist_id].clone() if wrist_id is not None else None
            wrist_q = robot.data.body_link_quat_w[0, wrist_id].clone() if wrist_id is not None else None
            # [v2] 指尖在腕局部系坐标：整手刚体漂移（wrist_3 零动作缓滚）在此自动抵消，只剩关节效应
            tips_local = {}
            if wrist_p is not None and wrist_q is not None:
                for nm in tip_names:
                    rel = (tips[nm] - wrist_p).unsqueeze(0)  # (1,3)
                    tips_local[nm] = quat_apply_inverse(wrist_q.unsqueeze(0), rel)[0].clone()
            out = dict(
                tips=tips,
                tips_local=tips_local,
                tip_q=tip_q,
                jpos=robot.data.joint_pos[0].clone(),
                jvel=robot.data.joint_vel[0].clone(),
                forces=mdp_obs.fingertip_contact_force(uw)[0].clone(),
                cube_p=cube.data.root_pos_w[0].clone(),
                cube_q=cube.data.root_quat_w[0].clone(),
                wrist_p=wrist_p,
                wrist_q=wrist_q,
                tcp_p=mdp_obs.tcp_position(uw)[0].clone(),
            )
        return out

    def fmt_vec(v, nd=3):
        return "(" + ",".join(f"{x:+.{nd}f}" for x in v.tolist()) + ")"

    # [v3] 四元数小工具（wxyz）——指尖姿态增量换算（自实现，免额外依赖）
    def _quat_mul(a, b):
        w1, x1, y1, z1 = a.unbind(-1)
        w2, x2, y2, z2 = b.unbind(-1)
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    def _quat_conj(a):
        w, x, y, z = a.unbind(-1)
        return torch.stack([w, -x, -y, -z], dim=-1)

    def _rot_delta_deg(q_tip_a, q_tip_b, q_wrist_a, q_wrist_b):
        """指尖相对手的姿态增量（腕局部系，度）。ΔP = conj(R_a)·(Q_a⊗conj(Q_b))·R_b 的旋转矢量。"""
        dp = _quat_mul(_quat_mul(_quat_conj(q_wrist_a), _quat_mul(q_tip_a, _quat_conj(q_tip_b))), q_wrist_b)
        if dp[0] < 0:
            dp = -dp
        w = dp[0].clamp(-1.0, 1.0)
        ang = 2.0 * torch.acos(w)
        if ang.item() < 1e-6:
            return torch.zeros(3, device=dp.device)
        s = torch.sqrt(torch.clamp(1.0 - w * w, min=1e-12))
        return (dp[1:] / s) * torch.rad2deg(ang)

    actions = torch.zeros(env.action_space.shape, device=dev)

    def hold(n: int):
        for _ in range(int(n)):
            with torch.inference_mode():
                env.step(actions)

    # ═════════════════ P0：零动作基线（从零动作点开始）═════════════════
    print("\n[P0] 零动作基线：本脚本从零动作点开始——不施加任何动作、不做接近/预设。")
    env.reset()
    s_reset = snapshot()  # reset 瞬间（物理落定前）
    print(f"[P0] reset 瞬间已记录（jvel max|v|={s_reset['jvel'].abs().max().item():.3f} rad/s）")

    hold(args_cli.settle_steps)  # 零动作 settle
    s0 = snapshot()
    print(f"[P0] 零动作 settle {args_cli.settle_steps} 步后：jvel max|v|={s0['jvel'].abs().max().item():.3f} rad/s"
          f"（应≈0；若>0.2 说明尚未稳，可加大 --settle_steps）")
    print(f"[P0] 与 reset 瞬间的最大关节差 = {(s0['jpos'] - s_reset['jpos']).abs().max().item():.4f} rad")

    print("[P0] — 关节位置(rad)：")
    for i, nm in enumerate(joint_names):
        print(f"        {nm:24s} {s0['jpos'][i].item():+.4f}")

    half = float(mdp_obs._CUBE_HALF_SIZE)
    print("[P0] — 五指指尖：世界坐标 | 面距(球近似, 与 _fingertip_cube_surface_dists 同口径) | 接触力：")
    for k, nm in enumerate(tip_names):
        d_center = torch.norm(s0["tips"][nm] - s0["cube_p"]).item()
        d_face = max(d_center - half, 0.0)
        print(f"        {nm:8s} {fmt_vec(s0['tips'][nm])}  面距={d_face:.3f} m  F={s0['forces'][k].item():.4f} N")
    print(f"[P0] — cube : pos={fmt_vec(s0['cube_p'])}  quat={fmt_vec(s0['cube_q'])}")
    print(f"[P0] — wrist: " + (fmt_vec(s0["wrist_p"]) if s0["wrist_p"] is not None else "n/a")
          + f"   TCP={fmt_vec(s0['tcp_p'])}")

    # ── 基线 CSV ─────────────────────────────────────────────────────────
    os.makedirs(args_cli.out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_csv = os.path.abspath(os.path.join(args_cli.out_dir, f"hand_probe_baseline_{ts}.csv"))
    with open(base_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "name", "v0", "v1", "v2", "v3"])
        for i, nm in enumerate(joint_names):
            w.writerow(["joint_rad", nm, f"{s0['jpos'][i].item():.6f}", "", "", ""])
        for nm in tip_names:
            t = s0["tips"][nm]
            w.writerow(["tip_world_m", nm, f"{t[0].item():.6f}", f"{t[1].item():.6f}", f"{t[2].item():.6f}", ""])
        for k, nm in enumerate(tip_names):
            w.writerow(["force_N", nm, f"{s0['forces'][k].item():.6f}", "", "", ""])
        cp, cq = s0["cube_p"], s0["cube_q"]
        w.writerow(["cube_pos_m", "cube_obj", f"{cp[0].item():.6f}", f"{cp[1].item():.6f}", f"{cp[2].item():.6f}", ""])
        w.writerow(["cube_quat_wxyz", "cube_obj", f"{cq[0].item():.6f}", f"{cq[1].item():.6f}", f"{cq[2].item():.6f}", f"{cq[3].item():.6f}"])
        if s0["wrist_p"] is not None:
            wp = s0["wrist_p"]
            w.writerow(["wrist_pos_m", "wrist_3_link", f"{wp[0].item():.6f}", f"{wp[1].item():.6f}", f"{wp[2].item():.6f}", ""])
        tp = s0["tcp_p"]
        w.writerow(["tcp_pos_m", "tcp", f"{tp[0].item():.6f}", f"{tp[1].item():.6f}", f"{tp[2].item():.6f}", ""])

    # ═════════════════ P1：8 通道逐维标定（±amp）═════════════════
    print(f"\n[P1] 通道逐维标定：一次一维、±{args_cli.amp} rad 对称，hold {args_cli.hold_steps} 步；臂保持零动作不动。")
    ch_rows = []
    for lbl, cidx in channels:
        for sign in (+1, -1):
            # 回零稳一手，取 before
            actions.zero_()
            hold(10)
            sb = snapshot()
            # 单通道命令
            actions.zero_()
            actions[:, cidx] = sign * args_cli.amp
            hold(args_cli.hold_steps)
            sa = snapshot()

            # 关节位移（取 top5、|Δ|>0.02 rad）
            dj = (sa["jpos"] - sb["jpos"])
            idx_sorted = torch.argsort(dj.abs(), descending=True).tolist()[:5]
            moved = [(joint_names[i], dj[i].item()) for i in idx_sorted if abs(dj[i].item()) > 0.02]
            moved_str = "; ".join(f"{n}:{d:+.3f}" for n, d in moved) if moved else "（无 >0.02rad 位移）"

            # 指尖位移（cm）：handΔ=腕局部系（干净，主判读）；worldΔ=世界系（含臂漂移污染，仅存档）
            tip_hand_strs, tip_world_strs = [], []
            for nm in tip_names:
                if sa["tips_local"] and sb["tips_local"]:
                    dh = (sa["tips_local"][nm] - sb["tips_local"][nm]) * 100.0
                    tip_hand_strs.append(f"{dh[0]:+.2f},{dh[1]:+.2f},{dh[2]:+.2f}|{torch.norm(dh).item():.2f}")
                else:
                    tip_hand_strs.append("n/a")
                dwt = (sa["tips"][nm] - sb["tips"][nm]) * 100.0
                tip_world_strs.append(f"{dwt[0]:+.2f},{dwt[1]:+.2f},{dwt[2]:+.2f}|{torch.norm(dwt).item():.2f}")

            # [v3] 指尖姿态增量（腕局部系，度）
            tip_rot_strs = []
            for nm in tip_names:
                if sa["wrist_q"] is not None and sb["wrist_q"] is not None:
                    rv = _rot_delta_deg(sa["tip_q"][nm], sb["tip_q"][nm], sa["wrist_q"], sb["wrist_q"])
                    tip_rot_strs.append(f"{rv[0]:+.1f},{rv[1]:+.1f},{rv[2]:+.1f}")
                else:
                    tip_rot_strs.append("n/a")

            fmax = sa["forces"].max().item()
            arm_d = (sa["jpos"][arm_joint_idx] - sb["jpos"][arm_joint_idx]).abs().max().item() if arm_joint_idx else 0.0
            warn = "   ⚠️ 力>0.05N：可能意外接触，请检查！" if fmax > 0.05 else ""
            print(f"      ch{cidx} {lbl} {sign:+d}x{args_cli.amp}: 关节[{moved_str}] 臂漂移={arm_d:.4f}rad")
            print(f"          handΔ(cm) " + " / ".join(f"{nm}:{s}" for nm, s in zip(tip_names, tip_hand_strs)) + warn)
            print(f"          rotΔ(deg) " + " / ".join(f"{nm}:({s})" for nm, s in zip(tip_names, tip_rot_strs)))

            ch_rows.append([lbl, cidx, f"{sign:+d}", args_cli.amp, moved_str]
                           + tip_hand_strs + tip_world_strs + tip_rot_strs + [f"{fmax:.4f}", f"{arm_d:.4f}"])

            # 回零
            actions.zero_()
            hold(15)

    # 收尾回到零动作
    actions.zero_()
    hold(20)
    s_end = snapshot()
    if arm_joint_idx:
        d_all = (s_end["jpos"] - s0["jpos"]).abs()
        worst_i = max(arm_joint_idx, key=lambda i: d_all[i].item())
        wrist_i = next((i for i, nm in enumerate(joint_names) if "wrist_3" in nm), None)
        w3 = f"；wrist_3 单看 {d_all[wrist_i].item():.4f} rad" if wrist_i is not None else ""
        print(f"\n[P1] 全程臂关节漂移（零动作=无恢复力，缓慢重力漂移属已知现象 2026-09-02）："
              f"max|Δ|={d_all[worst_i].item():.4f} rad @{joint_names[worst_i]}{w3}")

    ch_csv = os.path.abspath(os.path.join(args_cli.out_dir, f"hand_probe_channels_{ts}.csv"))
    with open(ch_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "channel_idx", "sign", "amp_rad", "moved_joints(rad)"]
                   + [f"{nm}_handΔ_cm|mag" for nm in tip_names]
                   + [f"{nm}_worldΔ_cm|mag" for nm in tip_names]
                   + [f"{nm}_rotΔ_deg_xyz" for nm in tip_names]
                   + ["max_force_N", "arm_drift_rad"])
        w.writerows(ch_rows)

    print(f"\n[PROBE] 完成。CSV:\n        基线: {base_csv}\n        通道: {ch_csv}")
    print("[PROBE] 解读提示：")
    print("  · 判读以 handΔ（腕局部系）为准——worldΔ 被零动作腕漂移污染，仅供参考。")
    print("  · 拇指通道：把 thumb4 位移方向朝向『绕到对侧面』的维度=两段式第一段候选；")
    print("    『朝掌内收/压向拇指侧』的维度=第二段（闭合贴面）候选——符号看 ± 两行对比。")
    print("  · four 通道：验证四指共享同步；各指位移幅度差=指长/关节差异的直接读数。")
    print("  · 把控制台文本或两份 CSV 发回，我据此设计 full 脚本的绕-合时序与预科角。")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
