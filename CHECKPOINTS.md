# UR5e 抓取项目 · Checkpoint 索引（2026-09-25 初版 / 2026-09-27 补充四指线）

> 用途：把 `logs/rsl_rl/ur5e_grasp_v5/best/` 里的"最佳节点"映射回原始 run 目录与 `model_<iter>.pt`，
> 并记录各自的战绩与所需配置。映射方法：读取 ckpt 内部 `iter` 字段 + 对原始文件做 md5 内容校验。

## 使用方式（play / 续训）

```bash
# play（GUI）
python scripts/rsl_rl/play.py --task Template-Ur5e-Drillgrasp-v0 --num_envs 16 \
  --checkpoint logs/rsl_rl/ur5e_grasp_v5/best/<best文件>.pt

# 续训（示例）
python scripts/rsl_rl/train.py --task Template-Ur5e-Drillgrasp-v0 --num_envs 2048 --headless \
  --resume --load_run <原始run目录名> --checkpoint model_<iter>.pt
```

## ① 旧预设线（v8.x：中指 70°/25° 预设、仅指尖碰撞、无幽灵 cube）

| best 文件 | 原始（run/model） | 战绩 | 备注 |
|---|---|---|---|
| `grasp_lift_best.pt`、`grasp_lift_v85_8980_final.pt` | `2026-09-22_02-29-22/model_8980.pt` | **99.9% 确定性达标、play 验收通过** | 旧资产"金标准" |
| `grasp_lift_v84_1550_success99.4pct.pt` | `2026-09-22_01-43-25/model_1550.pt` | 99.4% | v8.4 峰值 |
| `grasp_lift_v84_1600_success99.9pct.pt` | `2026-09-22_01-43-25/model_1600.pt` | 99.9% | v8.4 峰值 |
| `grasp_lift_v84_hold_liftS70.pt` | `2026-09-22_01-43-25/model_1375.pt` | 保持/提起稳定性 | v8.4 |
| `grasp_lift_v82_1085_peak_rew140_gate1.68.pt` | `2026-09-22_00-52-21/model_1085.pt` | gate 1.68 / rew 峰值 | v8.2 |
| `grasp_lift_v71_140.pt` | `2026-09-21_20-51-19/model_140.pt` | 早期里程碑 | — |
| `grasp_lift_v72_160_high6cm.pt` | `2026-09-21_20-59-20/model_160.pt` | 提起 6cm | — |
| `grasp_lift_v73_165_grip1.0.pt` | `2026-09-21_21-06-21/model_165.pt` | grip 1.0 | — |
| `grasp_lift_v73_1000_lift5.9cm.pt` | `2026-09-21_21-06-21/model_1000.pt` | 提起 5.9cm | — |
| `grasp_lift_v74_1035_lift8cm27pct.pt` | `2026-09-21_23-01-44/model_1035.pt` | 8cm / 27% | — |
| `grasp_lift_v75_1030_grip0.97.pt` | `2026-09-21_23-14-50/model_1030.pt` | grip 0.97 | — |
| `grasp_lift_v75_1035_lift9.7cm.pt` | `2026-09-21_23-14-50/model_1035.pt` | 9.7cm | — |
| `grasp_lift_v76_1045_lift9.6cm_28pct.pt` | `2026-09-21_23-27-57/model_1045.pt` | 9.6cm / 28% | — |
| `grasp_lift_v77_1055_grip0.98.pt` | `2026-09-21_23-40-46/model_1055.pt` | grip 0.98 | — |
| `grasp_lift_v77_1060_lift14.5cm_34pct.pt` | `2026-09-21_23-40-46/model_1060.pt` | 14.5cm / 34% | — |
| `grasp_lift_v78_1065_grip0.98.pt` | `2026-09-21_23-54-00/model_1065.pt` | grip 0.98 | — |
| `grasp_lift_v78_1070_gate1.0_lift41pct.pt` | `2026-09-21_23-54-00/model_1070.pt` | gate 1.0 / 41% | — |

⚠️ 该线只配旧资产（中指 70/25 预设 + 仅指尖碰撞 + 无幽灵 cube），当前资产下不可直接 play。
（`v79_1075_grip_max`、`v80_1080_...`、`new1000/old1000` 的原始文件已被覆盖，仅存 best 副本。）

## ② 路径延长线（v17：中指伸展 20°/20°、无幽灵 cube、仅指尖碰撞）

| best 文件 | 原始（run/model） | 战绩 |
|---|---|---|
| `grasp_v17_gate078_lift16mm.pt` | `2026-09-23_16-34-09/model_4335.pt` | gate 0.78 / 净提 1.6cm（早期里程碑） |
| `grasp_v17_lift38mm_succ37.pt` | `2026-09-23_17-11-38/model_4665.pt` | 净提 3.8cm / 37% |
| `grasp_v17_verified_208mm.pt` | `2026-09-23_17-11-38/model_6075.pt` | **确定性 +20.8cm、力 0.86/0.90N（该线最强）** |
| `grasp_v17_latest_best.pt` | `2026-09-23_21-54-06/model_6365.pt` | lift 12.0 / succ 3.6 |

## ③ 新碰撞线（v23+：中指 30/30°、拇指 -40/-80/+50/+40°、多段碰撞）

| best 文件 | 原始（run/model） | 战绩 | 配置要点 |
|---|---|---|---|
| `grasp_v23_nc_lift48mm.pt` | `2026-09-24_18-43-26/model_945.pt` | 新碰撞下首次捏提成功 | t2 冻结、face_reach 3.0、palm 1.4、近起 |
| `grasp_v23_nc_latest_best.pt` | `2026-09-24_21-50-48/model_2420.pt` | **确定性 +14.7cm、力 0.6/0.8N** | 同上 |
| `grasp_v24_retreat_3abc.pt` | `2026-09-24_23-01-53/model_3430.pt` | lift 13.2 / succ 4.2、**力 0.78/1.04N** | 3a-3c 退火：face_reach 0、palm 1.0、t2 解冻 |

## ④ 接近线（v26 cube 随机化 / v29 手臂远起）

| best 文件 | 原始（run/model） | 战绩 | 配置要点 |
|---|---|---|---|
| `grasp_v26_approach_1cm.pt` | `2026-09-25_01-19-25/model_4340.pt` | lift 13.3 / succ 86%、reach 1.19 | cube ±1cm 随机化、tcp_reach 1.5、palm 到位门控 |
| `grasp_v26_approach_2cm.pt` | `2026-09-25_01-44-46/model_5035.pt` | gate 1.6 / lift 11.3 / succ 3.6 | cube ±2cm 随机化 |
| `grasp_v29_approach_far015.pt` | `2026-09-25_16-50-10/model_1565.pt` | 全流程首通（远起 1.7cm） | 远起 scale 0.15、face_reach 3.0、t2 冻结 |
| `grasp_v29_far3.5cm.pt` | `2026-09-25_19-21-05/model_1885.pt` | lift 11.9 / succ 3.5 | 远起 scale 0.3（≈3.5cm） |
| **`grasp_v29_far6cm.pt`** | **`2026-09-25_19-51-28/model_2215.pt`** | **全流程最佳**：远起≈6cm → 接近 → 捏 0.5-0.7N → **提 +10.8cm**（确定性，仍上升） | 远起 scale 0.5、face_reach 3.0、t2 冻结 |

## ⑤ v30 xy 解锁线（中指+拇指；去掉 xy 锁定后）

| best 文件 | 原始（run/model） | 战绩 | 配置要点 |
|---|---|---|---|
| `grasp_v30_unlock_v1.pt` | `2026-09-25_23-48-46/model_2930.pt` | 首次"xy 自由也能提" | 删 cube_xy_clamp + cube_motion |
| `grasp_v30_unlock_v2_drift24mm.pt` | `2026-09-26_00-57-47/model_3655.pt` | 确定性 提+12.7cm / 漂移 2.4cm | 加 cube_drift -0.8 |
| `grasp_v30_unlock_v3_drift15mm.pt` | `2026-09-26_11-46-17/model_11700.pt` | 确定性 提+12.4cm / 漂移1.5cm、力0.78/1.04N | drift -1.5、lr 3e-5 |

## ⑥ v31 t2 自由 + 远起 8cm（中指+拇指）

| best 文件 | 原始（run/model） | 战绩 | 配置要点 |
|---|---|---|---|
| `grasp_v31_t2free_far8cm.pt` | `2026-09-26_12-38-13/model_13025.pt` | 确定性 捏0.68N / 提+9.5cm / 漂移1.9cm | t2 解冻、远起 scale 0.7 |

## ⑦ 四指共享+拇指线（当前方向；食/中/无/小 j2=j3=30°、四指全碰撞）

| best 文件 | 原始（run/model） | 战绩 | 配置要点 |
|---|---|---|---|
| `grasp_v34_4finger_v1.pt` | `2026-09-26_17-12-20/model_2125.pt` | 首次四指捏提（有"拖拽"漏洞，留档） | 近起、t2 冻结、xy 自由 |
| `grasp_v34b_4finger_drift_fix.pt` | `2026-09-26_20-26-23/model_2765.pt` | **确定性 提+14.2cm / 漂移≤2.8cm** | +漂移>4cm 终止、drift -1.5 |
| `grasp_v35_4finger_t2free.pt` | `2026-09-27_02-50-58/model_3010.pt` | 确定性 提+13.3cm / 漂移2.3cm | t2 解冻 |
| `grasp_v36_4finger_reach015.pt` | `2026-09-27_03-17-03/model_3315.pt` | 训练 lift13.0 / succ3.94 | 开 reach、远起 0.15 |
| `grasp_v36b_4finger_reach035.pt` | `2026-09-27_03-47-27/model_3665.pt` | lift11.7 / succ3.45 | 远起 0.3 |
| `grasp_v36c_4finger_reach06cm.pt` | `2026-09-27_04-22-35/model_4225.pt` | lift9.9 / succ3.1 | 远起 0.5 |
| `grasp_v37_slim_noreach06cm.pt` | `2026-09-27_16-06-30/model_11005.pt` | 瘦身生效：lift11.5 / succ3.76 | tip_progress=0 |
| `grasp_v37b_slim_far8cm.pt` | `2026-09-27_16-48-36/model_11375.pt` | lift11.6 / succ3.77 | 远起 0.7 |
| **`grasp_v37c_slim_far116cm.pt`** | **`2026-09-27_17-23-44/model_11715.pt`** | **四指线最佳**：确定性 12cm 远起→捏(拇0.8N)→提+13.4cm / 漂移≤1.8cm；训练 lift13.4 / succ4.36 | 远起 1.0 |
| `grasp_v38_4finger_multicol.pt` | `2026-09-27_18-34-05/model_12005.pt` | 食/无/小 2/3 段碰撞开启后微调：lift12.9 / succ4.18、确定性 提+13.4cm / 漂移≤1.8cm | 全碰撞 |

## ⑧ 自动课程线（v39~v48：代码内课程，一条命令从头跑到最终部署）

目标：把"人工接力课程"写进代码——一次 `train.py` 从随机权重跑到最终分布，全自动、可复现。
关键机制（`mdp/curriculum.py` + `ur5e_drillgrasp_env_cfg.py::CurriculumCfg`）：
- 信号：`lift_success` 的逐环境 EMA（α=0.001）；所有判定"阈值+连续 3 次确认"；退火期全程"退化即暂停"（EMA<0.35 冻结进度）。
- 易起脚手架（从头期；学会后自动退）：`reset_arm_prepose` 臂腕"接触预备位"（腕1 −0.28/腕2 −0.20 等）+ `curriculum_hand_preclose_anneal` 手部预闭合（改 default 关节位=动作目标，持久）。解决"直闭拢指尖差 1.4~2cm 够不到盒面"的从零探索死结。
- 远起阶梯 `arm_far`：0→0.15→0.3→0.5→0.7→1.0（EMA≥0.45、每档≥5000 步、只升不降、续训自动对齐）。
- 先松后严（EMA≥0.5）：`contact_gate.thr` 0.03→0.15；提门 `GATE_LIFT` 0.10/0.15→0.35/0.2；`cube_motion` 0→−0.8；`cube_drift` 0→−1.5。
- 串行退火链：soft unfreeze（t2 thaw 0→1，20000 步防跳变）→ `tip_progress` 12→0（12000）→ `tcp_reach` 1.5→0.5（12000）。
- `thumb_face_reach` **保留 3.0 不再退火**（实测退到 0 的瞬间策略崩：gate 1.23→0.036）。

| 阶段 | 事件 | 结论 |
|---|---|---|
| v39~v40 | k=1 门控+顺形；续训验证课程（梯对齐 4/4、face_reach/tcp 退火不塌） | 课程机制可用；修 `EventManager` 无 `_terms` 的 API 错 |
| v41~v44 | 从头跑 4 配方（t2 自由/冻结、tip 0/12、近起/远起、惩罚渐入）全部卡"悬停"（gate 0.02） | 现碰撞+严门控下"直接从零闭拢"探索不出去 → 需要易起起点 |
| v45h | 加预备位+预闭合后从头成功起飞（~2800 轮 pinch） | 方案有效；但 face_reach→0 崩、二值解冻崩 |
| **v48 清跑** | **`logs/train_v48_clean.log`（run `2026-09-29_16-48-51`）：16:48→01:31 ≈8.7h 全自动零干预，走完全链条** | 交付权重 `best/grasp_v48_clean_autofinal.pt`（=model_4520） |

**v48 确定性验收（100 集，最终部署配置：curriculum=None、预备位/预闭合=0、thr0.15、提门0.35/0.2、motion−0.8/drift−1.5、tip0、tcp0.5、face_reach3.0、t2 自由）**：
- 成功率 **100/100 = 100%**；提起 **mean 12.6cm / min 10.2cm**；
- 首次达 8cm 瞬间对向力：拇 **0.59N（min 0.50）**、四指 max **0.39N**；该瞬间 **xy 漂移 0.0cm**；
- 全程（含提起后保持期）xy 漂移 mean 3.3 / max 4.0cm（提起后滑动=此前用户定"先不管"项）。

> ⚠️ 部署/评测必须用"末态配置"（curriculum=None + 上列末值 + `frozen_channels=()`），不能用 cfg 文件里的"起始易值"（会分布错配，成功率骤降）。

## 当前建议起点（四指线）

- **主选（自动课程交付）**：`grasp_v48_clean_autofinal.pt`（= `2026-09-29_16-48-51/model_4520.pt`）——代码内全自动课程从零跑出、100% 验收。
- 四指线手动最佳：`grasp_v37c_slim_far116cm.pt`（= `2026-09-27_17-23-44/model_11715.pt`）——四指线验证最佳：12cm 远起 + 全流程 + 漂移≤1.8cm。
- **全碰撞版**：`grasp_v38_4finger_multicol.pt`（食/无/小 2/3 段碰撞开启后的微调版）。
- **对照**：`grasp_v34b_4finger_drift_fix.pt`（近起、确定性提起最高 +14.2cm）。

## 附：如何复现映射

```python
# 读取 best ckpt 的 iter，再去各 run 目录找同名 model_<iter>.pt 并 md5 校验
ck = torch.load("logs/rsl_rl/ur5e_grasp_v5/best/<file>.pt", map_location="cpu", weights_only=False)
print(ck["iter"])
```

---
_整理时间：2026-09-25（初版）+ 2026-09-27（补 v30~v38）· 映射经 md5 内容校验；`best/` 文件与原始文件内容一致（除注明"原始已覆盖"者）。_
