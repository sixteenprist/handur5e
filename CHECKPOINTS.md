# UR5e 抓取项目 · Checkpoint 索引（2026-09-25 整理）

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

## 当前建议起点（xy 解锁用）

- **主选**：`grasp_v29_far6cm.pt`（= `2026-09-25_19-51-28/model_2215.pt`）——与新资产/多段碰撞/预设完全匹配，且含"远起→接近→捏→提"全流程。
- **对照**：`grasp_v24_retreat_3abc.pt`（捏力最强，但无接近、t2 自由）。

## 附：如何复现映射

```python
# 读取 best ckpt 的 iter，再去各 run 目录找同名 model_<iter>.pt 并 md5 校验
ck = torch.load("logs/rsl_rl/ur5e_grasp_v5/best/<file>.pt", map_location="cpu", weights_only=False)
print(ck["iter"])
```

---
_整理时间：2026-09-25 · 映射经 md5 内容校验；`best/` 文件与原始文件内容一致（除注明"原始已覆盖"者）。_
