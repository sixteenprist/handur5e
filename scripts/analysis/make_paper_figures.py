"""论文图表生成（2026-10-18）。数据来源全部注释在下方（eval 表 / TB 曲线 / CHECKPOINTS）。

用法: python scripts/analysis/make_paper_figures.py
输出: docs/figures/*.png (300dpi) + docs/figures/DATA_SOURCES.md
"""
import glob, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(ROOT, "logs/rsl_rl/ur5e_grasp_v5")
OUT = os.path.join(ROOT, "docs/figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved", p)

def tb_curve(run_dirs, tag):
    """拼接多个 run 的 TB 标量（按 step 去重；后一个 run 覆盖接缝）。"""
    d = {}
    for rd in run_dirs:
        fs = glob.glob(os.path.join(LOG, rd, "events*"))
        if not fs:
            continue
        ea = EventAccumulator(sorted(fs)[-1])
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        for s in ea.Scalars(tag):
            d[s.step] = s.value
    xs = sorted(d)
    return np.array(xs), np.array([d[x] for x in xs])


def smooth(x, y, k=100):
    if len(y) < k:
        return x, y
    return x[k-1:], np.convolve(y, np.ones(k) / k, mode="valid")

def ablation_curves():
    t = open(os.path.join(ROOT, "logs/train_ablate_nocurr.log"), encoding="utf-8", errors="ignore").read()
    pat = re.compile(r"Learning iteration (\d+)/21790.*?(?=Learning iteration |\Z)", re.S)
    xs, ls, el = [], [], []
    for m in pat.finditer(t):
        seg = m.group(0)
        mm = re.search(r"Episode_Reward/lift_success: ([\d\.]+)", seg)
        me = re.search(r"Mean episode length: ([\d\.]+)", seg)
        if mm and me:
            xs.append(int(m.group(1))); ls.append(float(mm.group(1))); el.append(float(me.group(1)))
    return np.array(xs), np.array(ls), np.array(el)

# ─────────────────────────────────────────────────────────────────────────
# Fig 1. 延迟-成功率（v57 vs v58）+ 动作延迟
#   来源: logs/eval_deploy_full2.log（v57 表）; logs/eval_deploy_actdelay.log（v58 表）;
#         每次 100 集、确定性、最终部署配置。
# ─────────────────────────────────────────────────────────────────────────
delays = [0, 1, 2, 4]
v57 = [100, 85, 16, 4]
v58 = [100, 100, 100, 92]
actd_labels = ["act 1", "act 2", "act1+β0.5", "obs1+act1"]
actd_vals = [100, 100, 100, 100]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
x = np.arange(len(delays)); w = 0.36
axes[0].bar(x - w/2, v57, w, label="v57 (no delay training)", color="#9aa0a6")
axes[0].bar(x + w/2, v58, w, label="v58 (delay-robust)", color="#2f6db3")
axes[0].set_xticks(x, [str(d) for d in delays])
axes[0].set_xlabel("Observation delay (control steps, 33 ms/step)")
axes[0].set_ylabel("Success rate (%)")
axes[0].set_ylim(0, 108)
axes[0].set_title("(a) Observation delay")
for i, (a, b) in enumerate(zip(v57, v58)):
    axes[0].text(i - w/2, a + 1.5, str(a), ha="center", fontsize=8)
    axes[0].text(i + w/2, b + 1.5, str(b), ha="center", fontsize=8)
axes[0].legend(fontsize=8)
xb = np.arange(len(actd_labels))
axes[1].bar(xb, actd_vals, 0.5, color="#2f6db3")
axes[1].set_xticks(xb, actd_labels, fontsize=8)
axes[1].set_ylim(0, 108)
axes[1].set_ylabel("Success rate (%)")
axes[1].set_title("(b) Action delay (v58)")
for i, v in enumerate(actd_vals):
    axes[1].text(i, v + 1.5, str(v), ha="center", fontsize=8)
fig.tight_layout(); save(fig, "fig1_latency_success.png")

# ─────────────────────────────────────────────────────────────────────────
# Fig 2. 部署裕度矩阵热力图（v57 vs v58, 13 个共有场景 + v58 新增动作延迟）
#   来源: 同 Fig1 两张表。
# ─────────────────────────────────────────────────────────────────────────
scen = ["baseline", "filter β0.7", "filter β0.5", "filter β0.33", "obs delay 1", "obs delay 2",
        "obs delay 4", "sensor noise", "arm rate 0.5", "arm rate 1.5", "arm rate 2.0",
        "β0.5 + obs delay1", "β0.5+obs2+noise"]
m57 = [100, 100, 88, 10, 85, 16, 4, 100, 1, 100, 100, 10, 2]
m58 = [100, 100, 100, 100, 100, 100, 92, 100, 9, 100, 100, 100, 100]
mat = np.array([m57, m58])
fig, ax = plt.subplots(figsize=(11, 3.2))
im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
ax.set_xticks(range(len(scen)), scen, rotation=30, ha="right", fontsize=8)
ax.set_yticks([0, 1], ["v57", "v58"], fontsize=9)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=8,
                color="black")
ax.set_title("Deployment margin: success rate (%) over 100 deterministic episodes")
fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
fig.tight_layout(); save(fig, "fig2_margin_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────
# Fig 3. 动作低通 β 扫描：成功率 + 保持期手颤
#   来源: eval_deploy 表（β 扫描）；手颤 = 保持期手部关节速度 RMS (rad/s)。
# ─────────────────────────────────────────────────────────────────────────
betas = [1.0, 0.7, 0.5, 0.33]
s57 = [100, 100, 88, 10]; s58 = [100, 100, 100, 100]
t57 = [0.284, 0.099, 0.120, 0.137]; t58 = [0.331, 0.133, 0.114, 0.114]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].plot(betas, s57, "o--", label="v57", color="#9aa0a6")
axes[0].plot(betas, s58, "o-", label="v58", color="#2f6db3")
axes[0].invert_xaxis(); axes[0].set_xlabel("β (1.0 = no filter; smaller = smoother)")
axes[0].set_ylabel("Success rate (%)"); axes[0].set_ylim(0, 108)
axes[0].set_title("(a) Success vs action low-pass")
axes[0].legend(fontsize=8)
axes[1].plot(betas, t57, "o--", label="v57", color="#9aa0a6")
axes[1].plot(betas, t58, "o-", label="v58", color="#2f6db3")
axes[1].invert_xaxis(); axes[1].set_xlabel("β")
axes[1].set_ylabel("Hold-phase tremor (rad/s)")
axes[1].set_title("(b) Tremor vs action low-pass")
axes[1].legend(fontsize=8)
fig.tight_layout(); save(fig, "fig3_filter_sweep.png")

# ─────────────────────────────────────────────────────────────────────────
# Fig 4. 零样本位姿容差（v57）：成功率与保持期漂移 vs cube 偏移
#   来源: 位姿 sweep 表（±2/±3/±4cm, 100 集/档）；基准 0cm 取自验收表。
# ─────────────────────────────────────────────────────────────────────────
off = [0, 2, 3, 4]
succ = [100, 100, 100, 100]
drift_mean = [2.3, 2.8, 3.6, 4.7]
drift_max = [2.6, 4.3, 7.0, 8.8]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot(off, drift_mean, "o-", label="drift mean", color="#2f6db3")
ax.plot(off, drift_max, "s--", label="drift max", color="#b35b2f")
ax.set_xlabel("Cube placement offset (cm, x/y)")
ax.set_ylabel("In-episode drift (cm)")
ax.set_title("Zero-shot pose tolerance (v57): 100% success at all offsets")
ax.set_xticks(off); ax.legend(fontsize=8)
for o, s in zip(off, succ):
    ax.annotate(f"success {s}%", (o, 0.4), ha="center", fontsize=7, color="gray")
fig.tight_layout(); save(fig, "fig4_pose_tolerance.png")

# ─────────────────────────────────────────────────────────────────────────
# Fig 5. v57 从头训练曲线（0→21798，含 3 次断点续跑）
#   数据: TB 拼接 [2026-10-11_11-44-05, 10-12_03-31-38, 10-12_08-58-48, 10-12_21-27-34]
# ─────────────────────────────────────────────────────────────────────────
runs57 = ["2026-10-11_11-44-05", "2026-10-12_03-31-38", "2026-10-12_08-58-48", "2026-10-12_21-27-34"]
fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
for ax, tag, ylab in [(axes[0], "Train/mean_reward", "Mean reward"),
                      (axes[1], "Train/mean_episode_length", "Mean episode length (steps)"),
                      (axes[2], "Loss/value_function", "Value loss")]:
    xs, ys = tb_curve(runs57, tag)
    ax.plot(xs, ys, lw=1.0, color="#2f6db3")
    for seam in (8200, 10870, 17450):
        ax.axvline(seam, color="gray", ls=":", lw=0.8)
    ax.set_ylabel(ylab)
axes[2].set_xlabel("Training iteration (global; dotted = resume seams)")
axes[0].set_title("v57: from-scratch training with automatic curriculum")
fig.tight_layout(); save(fig, "fig5_train_v57.png")

# ─────────────────────────────────────────────────────────────────────────
# Fig 6. 延迟微调：成功案例 vs 崩坏案例
#   数据: TB 2026-10-15_02-01-11（v58 微调, 21790→23789）;
#         2026-10-15_01-51-24（首次尝试, 21851 前崩坏）
# ─────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for col, (rd, title) in enumerate([("2026-10-15_02-01-11", "v58 fine-tune (0-1 / 0-1 steps)"),
                                   ("2026-10-15_01-51-24", "Failed first attempt (0-2 / 0-1)")]):
    for row, (tag, ylab) in enumerate([("Train/mean_reward", "Mean reward"),
                                       ("Loss/value_function", "Value loss")]):
        xs, ys = tb_curve([rd], tag)
        axes[row, col].plot(xs, ys, lw=1.0, color="#2f6db3" if col == 0 else "#b35b2f")
        axes[row, col].set_title(title, fontsize=9)
        axes[row, col].set_ylabel(ylab)
        if row == 1:
            axes[row, col].set_xlabel("Iteration")
fig.tight_layout(); save(fig, "fig6_delay_finetune.png")

# ─────────────────────────────────────────────────────────────────────────
# Fig 7. 动作空间迭代对照（8/12/16 维；不同训练预算，见正文说明）
#   来源: CHECKPOINTS ⑩⑪⑨（v49/v52/v53 验收表）
# ─────────────────────────────────────────────────────────────────────────
dims = ["8-D\n(shared four)", "12-D\n(j2/j3)", "16-D\n(j1/j2/j3)"]
lift = [10.6, 10.9, 11.5]
f4 = [0.26, 0.52, 0.51]
fig, ax1 = plt.subplots(figsize=(5.8, 3.4))
x = np.arange(3)
b = ax1.bar(x - 0.18, lift, 0.34, color="#2f6db3", label="Lift height (cm)")
ax1.set_ylabel("Lift height (cm)", color="#2f6db3")
ax1.set_ylim(0, 14)
ax2 = ax1.twinx()
ax2.bar(x + 0.18, f4, 0.34, color="#b35b2f", label="Four-finger max force (N)")
ax2.set_ylabel("Four-finger max force (N)", color="#b35b2f")
ax2.set_ylim(0, 0.7)
ax1.set_xticks(x, dims, fontsize=8)
ax1.set_title("Action-space progression (separate training lines)")
for i, v in enumerate(lift):
    ax1.text(i - 0.18, v + 0.2, f"{v}", ha="center", fontsize=8)
for i, v in enumerate(f4):
    ax2.text(i + 0.18, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
fig.tight_layout(); save(fig, "fig7_action_space.png")

# ─────────────────────────────────────────────────────────────────────────
# Fig 8. 四指贴面距离（v57 验收探测，2026-10-13 记录）
# ─────────────────────────────────────────────────────────────────────────
tips = ["thumb", "index", "middle", "ring", "little"]
tipcm = [1.3, 0.2, 0.4, 0.0, 0.0]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.bar(tips, tipcm, color=["#2f6db3", "#4a90d9", "#6aa6e0", "#8abce8", "#a9d2f0"])
ax.set_ylabel("Fingertip-to-face distance (cm)")
ax.set_title("v57: all four fingers stay near cube faces (0 = contact)")
for i, v in enumerate(tipcm):
    ax.text(i, v + 0.04, f"{v:.1f}", ha="center", fontsize=8)
ax.set_ylim(0, 1.6)
fig.tight_layout(); save(fig, "fig8_fingertips.png")

# ─────────────────────────────────────────────────────────────────────────
# Fig 9. 自动课程链示意（依据 env_cfg.CurriculumCfg 的 after_term）
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.axis("off")
def box(x, y, txt, w=1.5, h=0.32, c="#eef3fb"):
    ax.add_patch(plt.Rectangle((x - w/2, y - h/2), w, h, fc=c, ec="#2f6db3", lw=1.0))
    ax.text(x, y, txt, ha="center", va="center", fontsize=8)
def arr(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#555555"))
# chain 1
box(1.2, 2.6, "indep_mix (α 0→1)"); box(3.6, 2.6, "flex_mix (0→0.5)"); arr(1.95, 2.6, 2.85, 2.6)
# chain 2
box(1.2, 1.7, "four_contact (0→1.5)"); box(3.6, 1.7, "four_pose_ref (0→1.0)")
box(6.0, 1.7, "lift_four (n 0→4)")
arr(1.95, 1.7, 2.85, 1.7); arr(4.35, 1.7, 5.25, 1.7)
# chain 3
box(1.2, 0.8, "t2_unfreeze (gate 0.6)"); box(3.6, 0.8, "tip_progress (12→0)")
box(6.0, 0.8, "tcp_reach (1.5→0)")
arr(1.95, 0.8, 2.85, 0.8); arr(4.35, 0.8, 5.25, 0.8)
box(8.2, 1.7, "parallel:\narm_far, prepose,\ngate_thr, preclose,\ngate_lift, cube_motion,\ncube_drift", w=2.4, h=1.15, c="#f6f1e8")
ax.text(0.1, 3.25, "Automatic curriculum chains (after_term dependencies)", fontsize=10, weight="bold")
ax.set_xlim(0, 9.6); ax.set_ylim(0.1, 3.5)
save(fig, "fig9_curriculum_chain.png")

# ─────────────────────────────────────────────────────────────────────────
# Fig 10. 课程消融：自动课程 vs 无课程（同预算 21790）
#   数据: TB 2026-10-18_11-42-10（无课程，已完成）；runs57（自动课程）
#   不稳定事件（虚线）：~13k / ~19k / ~21.77k 末段
# ─────────────────────────────────────────────────────────────────────────
xs_a, ls_a, el_a = ablation_curves()
ls_v57_x, ls_v57_y = tb_curve(runs57, "Episode_Reward/lift_success")
el_v57_x, el_v57_y = tb_curve(runs57, "Train/mean_episode_length")
if len(xs_a) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x1, y1 = smooth(ls_v57_x, ls_v57_y)
    x2, y2 = smooth(xs_a, ls_a)
    axes[0].plot(x1, y1, lw=1.1, color="#2f6db3", label="v57 (auto curriculum)")
    axes[0].plot(x2, y2, lw=1.1, color="#b35b2f", label="no curriculum (ablation)")
    for ev in (13000, 19300, 20100, 21770):
        axes[0].axvline(ev, color="gray", ls=":", lw=0.8)
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("lift_success term (100-iter smooth)")
    axes[0].set_title("(a) Comparable success term"); axes[0].legend(fontsize=7)
    x3, y3 = smooth(el_v57_x, el_v57_y)
    x4, y4 = smooth(xs_a, el_a)
    axes[1].plot(x3, y3, lw=1.1, color="#2f6db3")
    axes[1].plot(x4, y4, lw=1.1, color="#b35b2f")
    for ev in (13000, 19300, 20100, 21770):
        axes[1].axvline(ev, color="gray", ls=":", lw=0.8)
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Mean episode length (steps, smooth)")
    axes[1].set_title("(b) Episode length (dotted: instability events)")
    fig.tight_layout(); save(fig, "fig10_no_curriculum_vs_curriculum.png")
    print("  NOTE: ablation curves parsed from text log; v57 from stitched TB; smoothing=100")
else:
    print("fig10: 无可用数据")

# ─────────────────────────────────────────────────────────────────────────
# Fig 11. 延迟语义时序图(示意, D_obs=D_act=1)
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.6, 3.4)); ax.axis("off")
steps = [0, 1, 2]
x0 = 1.0; dx = 2.9
for k, t in enumerate(steps):
    x = x0 + k * dx
    ax.add_patch(plt.Rectangle((x, 2.5), 1.9, 0.55, fc="#eef3fb", ec="#2f6db3", lw=1.0))
    ax.text(x + 0.95, 2.775, f"true state s_{t}", ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((x, 0.85), 1.9, 0.55, fc="#f6f1e8", ec="#b35b2f", lw=1.0))
    ax.text(x + 0.95, 1.125, f"applied action", ha="center", va="center", fontsize=9)
    ax.text(x + 0.95, 1.0, f"a_{t-1} (D_act=1)", ha="center", va="center", fontsize=7.5)
    ax.add_patch(plt.Rectangle((x, 1.7), 1.9, 0.45, fc="#ffffff", ec="#666666", lw=0.8, ls="--"))
    ax.text(x + 0.95, 1.925, f"policy receives o_{t-1} (D_obs=1)", ha="center", va="center", fontsize=7.2)
ax.annotate("", xy=(x0 + dx + 0.2, 1.72), xytext=(x0 + 0.95, 2.48),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#2f6db3"))
ax.annotate("", xy=(x0 + dx + 0.2, 1.42), xytext=(x0 + dx - 0.0, 1.68),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#b35b2f"))
ax.text(x0, 3.35, "Delay semantics during training (per-episode sampling; D_obs ∈ {0,1}, D_act ∈ {0,1})",
        fontsize=10, weight="bold")
ax.set_xlim(0, 10.4); ax.set_ylim(0.5, 3.6)
save(fig, "fig11_delay_timing.png")

# ── 数据来源清单 ──
with open(os.path.join(OUT, "DATA_SOURCES.md"), "w", encoding="utf-8") as f:
    f.write("""# 图表数据来源（2026-10-18）

| 图 | 数据来源 |
|---|---|
| fig1 延迟-成功率 | logs/eval_deploy_full2.log（v57 表）、logs/eval_deploy_actdelay.log（v58 表）；100 集/档，确定性 |
| fig2 裕度热力图 | 同上两张表（13 个共有场景） |
| fig3 滤波扫描 | 同上（β=1.0/0.7/0.5/0.33；手颤=保持期手部关节速度 RMS） |
| fig4 位姿容差 | 位姿 sweep（±2/3/4cm，100 集/档）+ v57 验收表（0cm）；v57 权重 |
| fig5 v57 训练曲线 | TB 拼接：2026-10-11_11-44-05 → 10-12_03-31-38 → 10-12_08-58-48 → 10-12_21-27-34 |
| fig6 延迟微调对照 | TB：2026-10-15_02-01-11（v58）、2026-10-15_01-51-24（失败尝试） |
| fig7 动作空间迭代 | CHECKPOINTS ⑨⑩⑪（v49/v52/v53 验收表；训练预算不同，不可作严格同预算对比） |
| fig8 四指贴面 | 2026-10-13 验收探测记录（四尖盒距） |
| fig9 课程链 | env_cfg.CurriculumCfg 的 after_term 关系 |
| fig10 课程消融对比 | 无课程=logs/train_ablate_nocurr.log(文本解析,全 21790);自动课程=TB 拼接 runs57;100 迭代平滑;虚线=不稳定事件(约 13k/19.3k/20.1k/21.77k) |
| fig11 延迟时序 | 机制示意（无数据；对应 §4.2 语义） |
""")
print("DATA_SOURCES.md 已生成")
