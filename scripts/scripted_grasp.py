# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""硬编程链 v0.28：臂自标定 + PB' + [PZ] + PC2 复现（2行）+ [PC7] 双侧同框·转动伺服修正（手指例行 c/e2 + 拇指慢滑·微捏）+ PC3 [1~4]+观察窗 + 摩擦材质。

[2026-09-19 新建] 全链"接近→抓取→提起"第一步（本版只到"站到位"，不按不提）：
  - P0 零动作基线：reset 后记录（腕位姿/指尖/cube），并取姿态保持目标 q_hold。
  - PA 臂自标定：逐维小幅 nudge（6 维），实测 raw→腕位姿响应方向/符号（世界系）→ 符号表；
    同时验证 OSC 语义（raw×0.005 = 每步目标增量，README 依据：position_scale/orientation_scale=0.005）。
  - PB 接近伺服：闭环把"四指指尖均值"送到 cube **-y 面外侧 `--clearance` m、面心高度**；
    姿态保持（对抗已知的零动作腕漂移，2026-09-02）。
    护：任何指尖力>0.03N 或 cube 位移>4mm 立即停。
  - PH 末端保持 `--hold_end` 步（站姿伺服继续），便于在 GUI 里观察"四指指腹朝向 / 拇指位置"。
  - PR 报告：五指指尖在 cube 局部系的坐标 + 各面距 + 腕位姿 + 关节角 + 力 → 控制台 + CSV。

[v0.2 变更·依据 v0.1 首跑实测]
  ① 位置伺服是"速度型"接口：每步位移 ≈ 0.13 × lead（lead=raw×0.005）→ 4mm lead 只有 ~0.5mm/步，
     262mm 接近 260 步只走了一半。→ step_clip 4mm→10mm（raw 2.0=raw_clip 上限）≈ 1.3mm/步，max_steps→700。
  ② q_hold 原本取在 reset+3 步（瞬态）叠加腕自由滚动 → PB 起步 rotErr 29.5°。
     → 改为 settle 后取 q_hold；PA 每维加"反向 nudge 抵消"（零动作=顺从模式，不会自动回位）。
  ③ 新增 PC 手部原地标定：在到位姿态（腕位姿锁住）逐通道 nudge，量 5 指尖在【腕局部系 W】
     与【cube 局部系 C】的位移 —— probe 手图是 reset 姿态的，姿态一换方向全变，
     这是"四指按压 + 拇指绕-合"的唯一可靠规划依据。
  ④ 新增 PC2 拇指可达性预试：5 组组合通道动作，量 thumb4 指尖能否进入"+y 面外 1cm 交战区"
     ——拇指允许用指尖侧面接触（用户 2026-09-19），判据只看"能不能进区"，不要求指腹朝向。

[v0.3 变更·依据 v0.2 首跑实测]
  ① v0.2 实测：PA 期间零动作休息段白攒 38° 腕滚；下降末段（it169-177）腕滚"突然收敛"，
     恰在同时 cube 被推动 4.3mm（指尖力 0.000N——是非指尖部位擦的，传感器盲区）。
     → 措施：PA 休息段改"腕位姿保持"；新增 PB-A 高位姿态归位（rotErr<3° 再下降）；
     cubeΔ 早停阈值 4→2mm；停时打全指尖坐标（找"谁碰的"）。
  ② v0.2 实测：PC2 判决——拇指在本姿态下够不到对侧（最好 |Δzone|=6.8cm，x 向差 6.7~9.3cm）。
     结合"四指指腹=link 局部 -z"（用户 GUI 确认）→ 抓取方案改为：
     四指卷曲面压（卷曲方向=cube 系 -z，j2→0.8/j3→0.6 刚好把指尖送到面心高度）+ 拇指角顶。
  ③ 新增 PC3 合拢预演 ×2（原站位 / 臂-x）：卷曲→+y 按压 creep→拇指满幅合拢，全程力限幅，
     报告两侧接触是否成立——为 v0.4 提起测试定稿。

[v0.4 变更·用户策略指令 2026-09-19]
  用户：①只需拇指 + 中指夹取（其余三指碰撞已关）；②手腕/TCP 位置与角度允许略微调整。
  几何提醒（实测推算）：腕旋转不改变两指尖的相对几何（只随手关节变），拇指-中指尖天然相距 ~13-16cm，
  而 5cm cube 的两侧夹持只需 ~7-10cm 指尖间距 → 缺口 ~2-4cm 只能靠“臂平移 -x”补；
  补上后：中指落 -y 面偏 -x 侧（贴面/贴边均可）、拇指落 +x/+y 角区（用指尖侧面顶）→ 对角夹持（法向仍正对）。
  → PC3 重写为“拇指-中指夹取预演”：中指卷曲 → 拇指合拢 → 臂 +y 联合压紧（双侧接触判据）；
  → PC2 目标点从 +y 面心改为 +x/+y 角区（拇指实际可达域）；
  → 站位两遍：原站位 / 臂 -x（--rehearsal_x，默认 -30mm）——找“中指上 -y 面 + 拇指上角”的平衡点。
  注：四指共享通道卷曲会带动 index/ring/little（无碰撞，穿模无碍）。

[v0.5 变更·依据 v0.4 首跑实测]
  ① v0.4 实测大幅进展：拇指真实咬住 cube（合拢触面 Fthumb=1.16~1.38N）；但“无停顿合拢”
     超过桌面摩擦（~1.18N）把 cube 推了 3.3/11.3mm → 后续 creep 全被 cubeΔ 保护叫停。
     中指卷曲后停在面外 2.6cm（还差一口气没压上去）。PC2 满幅 |Δzone| 只剩 2.8-3.0cm（v0.2 时 6.8）。
  ② v0.5 重写 PC3 为分阶段柔性合拢：卷曲力停(触面即止) → 拇指合拢力停(0.08N) →
     双侧交替加力（缺力侧加：拇指通道 +0.05 / 臂 +y 移 1mm）到 0.25N/侧 → 微提 +3mm 探测；
     cubeΔ 基准改为本遍自己的 cube_ref（防 PB 微移污染）；新增 --squeeze_force。

[v0.6 变更·依据 v0.5 首跑实测]
  ① v0.5 实测：PB 在 tipz≈0.816（面心上方 4-5cm）被"非指尖部位"（疑似掌根/腕底）顶住——
     |e| 冻结 550+ 步（cubeΔ 只 1.1mm < 早停阈值，没触发保护），rotErr 被顶到 9°。
     → v0.6：下降加"高位地板"（--stop_z_rel，默认 0.055：tip 均值到 rel+0.055 即停、不再下压）
     + 受阻检测（50 步内 |e| 无 1mm 进展 → 停并打全指尖）。
  ② v0.5 实测：卷曲后中指停在面外 ~0.9cm（差一口气）；拇指合拢能从 t=0.67 轻触 0.11N ✓（原站位）；
     但"拇指已接触后臂再 +y 爬"会把拇指楔进角部 → 力爆 0.88N、cube 被顶
     （-x 站位更糟：0.47N→1.18N、cube 被推 5.3cm）。→ v0.6 PC3 重排：
     **先中指落位（臂 +y，拇指还没合拢——零干扰）→ 再拇指合拢（力停 0.12N/cubeΔ 2mm、每 2 步采样）→ 微提**。

[v0.7 变更·依据 v0.6 首跑实测]
  ① v0.6 实测：PB 高位地板完美（136 步收敛、e_xy=2mm、rotErr 0.3°、cubeΔ=0，卡死消失）；PC2 拇指最远
     够到 +x/+y 角区外 3.7cm；但两遍预演的"+y 落位"都在 +2~4mm 就被顶住（Fmid=0.0000），cube 被推/被抬
     （原站位 +5.4mm z——被掀翻）。复算几何：中指尖**参考点**距面 14mm 时，真实表面已在参考点前 ~12mm
     （只剩 2~4mm 间隙）→ 爬 2~4mm 必触、且触的是无传感器指侧（Fmid 读 0 不代表没碰）。
  ② v0.7 夹取序列重排（拇指先行）：
     [1] 臂 -x 预置（--stage_x，默认 -18mm：把拇指让到 +x 侧贴面位、中指尖摆到面缘内侧）；
     [2] 拇指空中合拢；
     [3] 臂 -y 慢推 ≤18mm → 拇指前缘侧贴 +x/+y 区（其 -y 前缘面压 +y 面 +x 缘条，法向 -y）力停
         --thumb_touch=0.10N；
     [4] 中指卷到底 j2/j3=1.0（参考点再向面心扫 15~20mm）→ 双侧压紧判据 Fthumb > --squeeze_force=0.30N
         （cube 被中指顶入拇指 = 力平衡即"握持成立"）；
     [5] 若不够力：+y 细爬 0.5mm×5 步补压；[6] 微提 +2mm 判决 + 保持 300 步观察（GUI）；[7] 松开回收。
  ③ 全程打印：mid4 盒距（指尖参考点→cube 面；≈12mm=表面贴合，v0.6 标定）+ Fmid + Fthumb + cubeΔ。

[v0.8 变更·依据 v0.7 首跑实测]
  ① v0.7 实测：PB 稳（128 步、rotErr 0.3°）；PC2 拇指最远 3.9cm；但"-18mm 预置下空中合拢其实不空"——合拢途中
     拇指已触到 +x/+y 角区：粗暴采样（每 5 步 + dcb 0.8mm）下 cube 被推 4.2mm、Fthumb 冲到 0.23N（事后又回落=
     被掀起又落回，[5] 末 cubeΔ 只剩 0.3mm）；[4] 又被"陈旧位移 5.8mm"直接叫停 → 中指根本没扫入。
  ② v0.8 修正三件事：
     a. [2] 合拢改"力停"：每 2 步采样，Fthumb>0.8×thumb_touch 即停（保留触面时的合拢量），dcb>0.6mm 兜底；
     b. 每阶段后**重新基线** cube_ref（防"陈旧位移"误触守卫），[6] 同时打"总Δ"；
     c. [4]/[5] 的 cubeΔ 上限放宽到 3.5mm（压紧本来会把 cube 顶进拇指 1~3mm），并加 [5b] "拇指续合"
        （从中断处每 +0.02 续合）——压紧差一口气时可补。
  ③ 预期链路：合拢触面(≈0.1N，停) → 中指卷到底把 cube 顶入拇指 → Fthumb 升过 0.30 = ★压紧成立 → 微提。

[v0.9 变更·依据 v0.8 首跑实测]
  ① v0.8 大突破：力停合拢成功（[2] t=0.57 触面，Fthumb=0.27N，cubeΔ=0.0mm 没碰动！）；
     **[4] 中指卷到底形成了真正的双侧捏持**：Fmid 传感器首次醒来（0.128→0.308N）、Fthumb=0.23N、cubeΔ=0.0mm。
     但 [5] 的"+y 补压"把好局毁了：cube 被顶起 3.8mm（中心 +2.6mm z = 被掀斜）、Fthumb 衰减到 0.05，
     微提 dz=-0.07mm 未提起。⇒ **教训：捏持一旦成立，不许再平移臂；加力只能用"拇指续合"。**
  ② v0.9 [5] 重写为压紧定稿：Stage A 拇指续合（从中断 t 每 +0.015，力停 0.30N / cubeΔ 2mm）；
     Stage B（仍不够时）臂 +y 微爬 0.25mm 级（cubeΔ>1.5mm 即停）。[6] 微提改 5×0.5mm 慢提、每步打印 cube_z。
  ③ [2]/[4] 保持 v0.8 不变（已实测：干净触面 + 干净捏持）。

[v0.10 变更·依据 v0.9 首跑实测]
  ① v0.9 实测：**[4] 直接打出 ★双侧压紧成立**（t=1.00：Fthumb=0.357N、Fmid=0.317N、cubeΔ=0.5mm）；
     [6] 微提 5×0.5mm 慢提仍未提起（cube_z 仅 +0.03mm）——Fthumb 静置中衰减 0.36→0.24、提程中再降到 0.13：
     摩擦预算 μ(Fmid+Fthumb)≈0.7~0.8N ≈ 临界 mg 0.78N ⇒ **就差最后 0.1~0.2N 余量**。
  ② v0.10：a) 新增 --squeeze_top（0.45N）：[5A] 续合补压到 0.45N（不再到 0.30 就收工）；
     b) [6] 静置 15→8 步，并加 [6b] 起飞前收尾补压（每次 +0.01，至多 3 档）；c) 微提/观察不变。
     预期：ΣN≈0.85+ ⇒ μΣN≈1.2N vs mg 0.78N（≈1.5× 余量）→ 提起。

[v0.11 变更·依据 v0.10 首跑实测]
  ① v0.10 实测：**[5A] 补压大成功**——Fthumb 0.08→0.52N 而 cubeΔ=0.0mm（零扰动加力！）；但微提 +2.5mm 时
     cube_z 纹丝不动（-0.07mm），双力 0.3/0.45N 却提不起 ⇒ **捏持的摩擦预算不够**：提程中手指在滑、cube 不动
     = 静摩擦余额 < mg 0.78N，即指尖-方块有效摩擦远低于假设的 1.5。
  ② v0.11：a) --squeeze_top 0.45→**0.90N**（[5A] 迭代 10→16、Fmid 上限 0.6→0.9）——把 ΣN 抬到 ~1.3-1.5N；
     b) [6b] 起飞补压门槛 0.32→0.55；c) 提程每 2 档"保压"（Fthumb<0.5 时 +0.008 续合）；
     d) 提程打印加 wrist_z（验证手确实在上升）。
  ③ 若这次仍在某个 Fthumb 处被 cubeΔ 守卫叫停（cube 被推走）= 实测到"夹持上限"；若提起来 = 收工。

[v0.12 变更·依据 v0.11 首跑实测]
  ① v0.11 实测：**[5A] 补压到 Fthumb=1.02N 零扰动**（cubeΔ≤0.2mm）；手确实在升（wrist_z +3.2mm），
     提程双力 0.4/0.72N，**cube 仍纹丝不动** ⇒ 物理结论：静摩擦余额不足（抬升摩擦能力 ≈ μΣN 不够），
     根因两候选：a) 指尖-方块有效摩擦低（材质问题）；b) 拇指角接触的"斜下压"分量把 cube 压向桌面
     （抬升还需克服下压力）→ 抬高 μ 同时改善两候选。
  ② v0.12：a) 新增 [MAT] 段：**打印手部指尖/腕 + cube 的当前物理材质与摩擦系数**（绑定前基线）；
     b) 新增 --hand_friction（默认 2.0，0=关）：给机器人全部碰撞体绑定高摩擦材质（UsdPhysics.MaterialAPI
     + 物理绑定；失败不影响流程）；c) 其余流程（[2]力停合拢/[4]捏持/[5A]补压到 0.9/[6]慢提）保持不变。
  ③ 预期：μ↑ 后摩擦预算 ≈2N >> 需求 ≈1.4-1.8N → 应能提起；若仍不提起，[MAT] 输出将指出真实摩擦值，
     下一步转"几何：降低拇指下压分量"。

[v0.13 变更·用户指令 2026-09-19（落点重构）]
  ① 用户指出（GUI 实拍）：之前 PB 把"五指指尖均值"送到 -y 面外侧（侧向塞入），导致**手掌根本没落在
     cube 正上方**（掌根吊在侧后方）——拇指位置被站位锁死，永远够不到对侧面。正确做法：**TCP 的 xy
     对中 cube 质心 xy**，从正上方罩下去。
  ② v0.13 重写 PB'：[1] 高位 xy 对中（伺服 TCP.xy → cube 质心 xy，保持高度）；[2] 垂直下降，
     **首触即停**（thumb4/middle4 任一力 >0.05N，或 TCP 到达 cube 顶+5mm 地板，或 cube 被推动）；
     全程打 TCP/双指力/thumb4 相对位移。PB-A 姿态归位保留；PH 改为腕位姿保持（旧 P1 伺服作废）。
  ③ --stage_x 默认 -0.018→**0.0**（旧"给拇指让角"理由在正上方罩下时不成立）；
     --clearance / --stop_z_rel 标记为旧参数（PB' 不再使用）。
  ④ PC/PC2 保留（在新站位重新量手图与拇指可达性）；PC3 链路保留（防守卫不变——先看新站位下触到哪）。

[v0.14 变更·依据 v0.13 首跑实测（用户 GUI 指出"指尖没到 cube 就弯曲/合拢"）]
  ① v0.13 实测两个问题：
     a) 下降是"龟速"：增量目标 1.5mm/步 → 实测速度仅 ~0.18mm/步（速度≈0.13×lead 的已知特性），
        400 步上限只走 ~73mm，而需要 ~160mm 才到地板保护 → 手停在 cube 上方 ~9cm，
        之后 PC/PC2/PC3 全部在半空执行（这就是"没到位就合拢"的真因）。
     b) 下降途中还有未指令的 +x 侧滑 ~12cm（thumb4Δx 从 +102mm 长到 +213mm；锁定位 TCP 偏 cube +116mm）——
        疑因姿态伺服亚度级误差持续给微指令 + 位置轴零指令时的顺从漂移。
  ② v0.14 修正：
     a) [PB'-z] 重写为两段速：快段 lead 9mm（≈1.0mm/步）→ 最低指尖进入 cube 顶 +5.5cm 后转慢段 lead 2.5mm
        （≈0.33mm/步）；全程 xy 主动保持（TCP 误差伺服、±6mm/步钳幅）；预算 400→800 步；
        停止判据改为"最低指尖 ≤ cube 顶 +1cm"（第一次把手真正送到 cube 高度）；守卫：任意指尖力>0.05N / cubeΔ>2mm。
     b) 姿态伺服（hold_wrist_step / _servo_step）加 0.86° 死区 + raw 钳幅 1.5→1.0——防"亚度级空转"累积（疑似侧滑源）。
     c) 新增 [PZ] 站位诊断：五指 + wrist 的 cube 局部坐标、四指簇心、拇指-中指口心/口距——为"最后几厘米"对位提供实测。
     d) 新增 [PZ+1] 四指簇对准（--engage_y 默认开）：慢速滑动把四指簇心送到 -y 面外 1.6cm（12mm 表面标定 + 4mm 余量），
        守卫 Fmax>0.03N / cubeΔ>2mm / 220 步上限；已有接触自动跳过；对准成功后 [3]（-y 慢推）自动跳过。
     e) [MAT] 探查的 ComputeBoundMaterial() 返回值长度兼容修复（v0.13 跑报 unpack 错；摩擦材质绑定本身已成功：32 prim @ μ=2.0）。

[v0.15 变更·依据 v0.14 首跑实测（用户指令：TCP 必须刚好在 cube 质心、不要 y 向偏移）]
  ① v0.14 实测：
     a) 漂移修复生效：TCP_rel_cube=(+4,0,-31)mm——xy 只差 4mm（v0.13 是 +116mm）；但 z 停在质心下方 31mm
        （下降停止判据用的是"指尖高度"而非 TCP）。
     b) [PZ+1] 四指簇对准把整手推了 +62mm y——用户看到的"手掌 y 向偏移"+"手太近"就是它。
     c) PB-A 未收敛：rotErr 15.3°→12.8° 耗尽 300 步（v0.14 把姿态权限砍到 1.0+死区太狠），下降全程带 ~13° 倾斜。
     d) 关键测量（[PZ]）：TCP xy=质心、指尖顶+1cm 时——四指尖在 cube 前方 9~11cm、拇指在 +x 侧 10cm、
        口距 16.2cm；任何指头离 cube 最近 >6cm。⇒ TCP 钉在质心时不可能接触（手型几何，非 bug）；
        接触必须靠"从干净站位出发的接近动作"（v0.16 主题）。
     e) [4] 卷曲在近距把 cube 推走 4.9cm（Fmid=0：手指侧面推的，传感器盲区）；"★提起成功"是 cube 被推倒的
        回弹噪声（全程零接触力）——误报。
  ② v0.15 修正：
     a) 下降停止判据改为 **TCP → cube 质心（xyz）**：dzTCP≤0 才停（慢段 0.33mm/步逼近）；预算 800→900 步；
        下降终点比 v0.14 高 31mm（不再"太近"）。
     b) [PZ+1] 四指簇对准默认关闭（--engage_y 默认 1→0）——保留开关供以后试验。
     c) 姿态权限恢复：hold_wrist_step raw 钳幅 1.0→2.0（PB-A 要能拉回 15° 残差）、死区 0.015→0.010；
        PB-A 预算 300→800（不足会打 ⚠️）。
     d) P0 增"主动稳姿"40 步再锁 q_hold——减小 settle 瞬态/顺从漂移对姿态参考的影响（提升运行间可比性）。
     e) [6] 微提判决要求"有接触力"才认提起（修正 ①e 误报）。

[v0.16 变更·依据 v0.15 首跑实测（用户问：拇指探索为何一直停在 +x 邻侧？）]
  ① v0.15 实测：
     a) 干净站位成功：PB-A rotErr=2.2°@0 步；[PB'-z] 265 步到 dzTCP=-0.3mm；TCP_rel_cube=(+2,-0,-0)mm
        （用户确认"手掌位置到位"）；[PZ] 与 v0.14 几乎一致（站位可复现 ✓）。
     b) PC3 关键实测：[2] 拇指合拢**真的碰到了 cube**（t=0.83，Fthumb=0.169N，cubeΔ=0.1mm）——但那是 +x 邻侧；
        [4] 四指卷到底 Fmid 仍 0（指尖盒距还差 ~6cm）→ "★双侧压紧"是误报；微提时 cube_z 跳 +10mm 是拇指把 cube
        **撬起**的噪声（保持段又落回 0.775）→ "★提起成功"也是误报。
     c) 用户观察：拇指探索全在 +x 邻侧、"关节 2 只往一个方向弯曲"——查 URDF：拇指四关节限位均 ±3.14
        （不是机械限制；是我们所有合拢组合都只用了 t2 负向）。→ 本轮把拇指双向满幅测一遍。
  ② v0.16：
     a) PC2 扩展为**拇指双向探索**（t1±/t2±/t3± 逐通道 + 正/负合拢对比，8 组），每组打印拇指尖位置 +
        指腹朝向——用数据回答"拇指能不能离开 +x 邻侧、哪个方向能让指腹朝向 cube"。
     b) 新增 **[B2] 口心对位接近**（--mouth_align 默认开）：先原地合拢（[2]/[4]），实测"拇指尖-中指尖"
        中点（闭合口心）→ 沿"口心→质心"方向慢速平移（含 1cm 预压、上限 10cm；平移前拇指先松到 ~0.66 档
        防拖拽；守卫 Fmax>0.10N / cubeΔ>1.5mm）——把 cube 套进闭合口：**四指贴前面、拇指到后面（对侧）**；
        随后 [5A] 合拇指压紧。注：这步会平移 ~6-7cm（TCP 随之离开质心）——运输站位保持 TCP=质心，抓取必须"对位"。
     c) 判定修正：[4] "双侧压紧"需同时 Fmid>0.03；[6] "提起成功"需**双侧都有接触力**，并新增保持段末判定
        （保持 300 步后 cube 仍抬高才算真提起，防"撬一下"误报）。

[v0.17 变更·用户指令 2026-09-19（"姿势还不对，先别着急加压；t2 另一个方向有可能形成对侧"）]
  ① v0.16 实测：
     a) PC2 双向探索**证实用户直觉**：合拢B（t2 正向）是全表最优——拇指尖到 (+5.5,+3.1,+1.1)cm、
        离目标角区 |Δ|=4.5cm（此前最好 6.2~6.8）、且指腹 x 分量首次转负（-0.38，开始朝 -x 侧）；
        而 t2 负向的合拢A 指腹仍朝 +x（+0.44）。逐通道：t1+ / t2+ 是"把拇指往 -x 转"的两大主力。
     b) [B2] 只平移 4 步就停（Fmax 0.150N）：滑降方向让卷曲四指尖先蹭到 cube 顶前缘；
        [5A]/[5B] 一串 0.15~0.9N 读数是拇指在角上蹭来蹭去（cube 被推 2.4mm）→ 姿势未定前按了也白按。
     c) 用户指令：姿势先行、不要急着加压。
  ② v0.17（姿势专项）：
     a) PC2 改**t2 正向家族细扫** 8 组（t4=-1/0/+1 转指腹、t3=+1/.4、t1=.5、合拢A 对照），
        每组打印：指尖位置/Δzone/指腹朝向 + **拇指四关节实际角度**（看 raw=1 用掉多少行程、还剩多少）；
        停留加长到 4×cal_hold（方便 GUI 观察）。
     b) PC3 保留 [1][2][3][4]（现状合拢，空中）+ **150 步观察窗**（打 thumb4/mid4/指腹/Fmax/cubeΔ）——
        不加压、不提起；[B2] 与加压/微提段由 `--mouth_align`（默认 0）/`--press`（默认 0）门控，代码保留。

[v0.18 变更·依据 v0.17 首跑实测（j= 揭示"单方向"真相 + 合拢B 其实扫中了 cube）]
  ① v0.17 实锤：
     a) j=（拇指四关节实际角度）说清"只往一个方向弯曲"的真相：thumb1 只能负向转（限位约 [-90°,0°]，
        只有"指令正"才生效）；thumb3/thumb4 只能正向弯（下限=0°，负命令完全无效）——用户 GUI 看到的就是
        这两个；thumb2 双向都正常（j 到过 ±0.96）。且 raw 1.0 = 1.0 rad，thumb1 离限位还差 ~0.57 rad、
        t3/t4 上限未知 → 此前所有命令都没用满行程。
     b) 合拢B（t2+）闭合路径**真的扫进了 cube 角区**：首行 Fmax=0.268N、把 cube 推了 11.4mm（−x 向）；
        其后几行"没碰到"其实是 cube 已被推走（末行再推 19mm、累计 26.7mm）——PC2 数据被推挤污染。
     c) PC3 [3] −y 推 13.5mm 时拇指抓住了（被推走的）cube（F=0.517N）；[4] 卷曲又把 cube 推 7.2mm。
  ② v0.18（深弯探索 + 数据净化）：
     a) 新增 `--hand_clip`（默认 1.6）：运行时放宽手部 raw 限幅（只设在本脚本的 env_cfg 上——clip 是本
        环境自带的"探索安全阀"，训练代码不动）→ 可把 thumb 各通道命令到 1.5，把"深弯"一格一格试出来。
     b) PC2 改**深弯族** 8 行（复现合拢B / 深弯 A-E / 中深 / 合拢A 对照），打印 盒距 + 指腹朝向 + j=。
     c) 新增 `restore_cube()`：PC2 每行前 + PC3 前把 cube 复位到出生位（防串行污染；cubeΔ 因此变成
        "本行自己的推挤量"）。

[v0.19 变更·依据 v0.18 首跑实测（深弯成功、最接近 17mm、restore 失效教训）]
  ① v0.18 实锤：
     a) **深弯通道全部打通**：j= 实测 t1 −1.47~−1.50（限位约 −1.57）、t2 −0.97~−1.27、t3 +1.13~+1.48、
        t4 +1.11~+1.49 —— raw 1.5 全部落在真实关节行程内（此前只用到 ~1.0）；深弯闭合时拇指真的压得很实
        （Fmax 1.51N / 1.18N 两行）。
     b) **最优方向 = "t1 深 + t3≈1.0 + t4 深"**：深弯D（1.5,1,1,1.5）指尖到 (+3.8,+3.5,+3.0)cm、盒距 17mm
        （历史最接近；合拢B 24mm；中深 t3=.4 反而飞到 76mm）。规律：t3 不是越大越好（≈1.0 最佳）、
        t4 基本只"转指腹"、t1 深主要带 −x/降 z。但**闭合弧线本身仍到不了背面**（D 停在"右后上角外"，
        x 仍 +3.8 在 +x 面外、z +3.0 在顶面上）——结论同上一轮推理：**差的是"手整体挪 2~3cm"，不是手指行程**。
     c) **restore_cube() 失效**（报 "Inplace update to inference tensor outside InferenceMode"——仿真缓冲
        曾在 inference 语境下创建，写入必须同样包进 inference_mode）→ PC2 又带推挤污染（cube 先被推 11.3mm、
        再逐行回落），PC3 [2] 还被"cube 落稳的位移"误触发（t=0.03 即停）、[3]/[4] 几乎空跑。
  ② v0.19（修复 + 贴背实验）：
     a) restore_cube() 修复（写入包 `torch.inference_mode()`）；PC3 复位后加 15 步静置（防落稳位移误触守卫）。
     b) PC2 换**深弯精扫 8 行**（围绕 D：t2 1.0/1.2/1.5、t1 1.55、t3 1.2/.9、t4 1.6、合拢B 对照）。
     c) 新增 **[PC4] 深姿贴背滑移**（--pc4 默认 1）：对两个候选终姿（D / 合拢B）各做一次——
        摆好姿势后**手指完全不动**，手整体沿"指尖→背面目标点"方向 1mm/循环慢滑（上限 ~4.5cm），
        Fmax>0.10N（触到）或 cubeΔ>2mm（推走）停 → 观察 40 步 + 报接触几何。**直接回答"拇指能否贴到对侧"**。

[v0.20 变更·依据 v0.19 首跑实测（贴角"干净触到" + 四指也还没解锁行程）]
  ① v0.19 实锤：
     a) **restore_cube 修复生效**（无报错）——PC2 各行独立：D 复现第 1 行把 cube 推了 12.6mm（D 系
        闭合路径本来就贴着"顶右后角"走）、D2 推 4.0mm、D3~D7 只擦过（≤0.8mm）——**D 系闭合一直在碰撞边界上**。
     b) **[PC4] 合拢B 滑移 = 首次"干净触到"**：滑 10mm 时拇指尖传感器 Fth=0.228N、**cubeΔ=0.0mm**——
        从右侧滑入贴住右后角、**顶住而不推走**（此前所有接触要么推走要么压翘）。而 D 滑移能钻更深
        （盒距 9mm）但压的是**顶面角**（把 cube 压翘 4.3mm 又弹回、指尖传感器还没读数）→ 排除"压顶"路线。
     c) PC3：[2] 合拢触面恢复到干净水平（t=0.93、0.228N、cubeΔ=0.3mm），但 [4] 四指一卷就把 cube
        拱走（4.1mm 停）→ "t2− 合拢 + 四指卷"组合仍不成立。
  ② v0.20（三个针对性动作）：
     a) PC2 精简 3 行（D复现/D3/合拢B）——只做复现误差核对。
     b) [PC4] 触后加**加压段**（0.25mm/循环，Fmax>0.40N 或 cubeΔ>1mm 停）——直接量"贴住了能顶多重"。
     c) 新增 **[PC5] 深卷四指 + 双侧合围预试**（--pc5 默认 1）：**四指和拇指一样一直只卷到 1.0**（同款
        "行程没用满"）——本段把 j2/j3 渐进深卷到 1.4（解锁行程）→ ①量 mid4 到底能够多近（能否碰前面）
        ②叠拇指（D→合拢B）看"双侧同时到位"有没有力 ③最后慢滑 −x 看能否**夹住**（力升而 cubeΔ 不涨）。

[v0.21 变更·依据 v0.20 首跑实测（手指可达性解决 + 双侧"干净触到"都齐了 → 首次组装"合围"）]
  ① v0.20 实锤：
     a) **手指可达性解决了**：四指 j2/j3 深卷解锁后，j2 实测到 +1.38（行程真实存在），中指尖到
        (+2.4,−4.6,+2.4)cm、盒距 21mm——**首次干净碰到 cube 前面**（Fmid=0.206N、cubeΔ=0.0mm）！
        （此前 j2/j3 只用到 1.0，中指尖一直差前面 2~5cm。）
     b) 拇指侧：合拢B 滑入 14mm 干净触到（Fth=0.129→0.23N、cubeΔ=0.0）；触后加压能维持 ~0.2N 后
        沿棱边滑脱衰减——**"轻触稳、深压滑"**（斜擦棱角的接触几何问题，不是力不够）。D 姿滑移 27mm 后
        也触到（Fth=0.74N）但压的是顶面 → 压顶路线弃用。
     c) PC5b：手指深卷在位时，**拇指 D 闭合路径又把 cube 推了 12.4mm**——组装序列禁用 D；PC5c 一滑
        就报警（ring4 蹭到 0.572N）——手指在前时，侧面滑入要更保守。
     d) PC3 现状：合拢 t=1.00 才停（盒距 37mm 没触）；[3] 臂 −y 3mm 时拇指压到 0.855N（零位移）；
        [4] 手指一卷把 cube 拱走 8.7mm——旧组合（合拢+卷指）退役。
  ② v0.21（首次组装"合围" = [PC6]）：
     手指深卷到位（停在首次触到 Fmid>0.15）→ 拇指合拢B + 滑入到位（Fth>0.10）→ **收口**（手指加深
     j2≤+0.3 + 臂 +y 细进，Fmax>0.60N / cubeΔ>1.5mm 停）→ **微提 +1mm 探测**（cube 跟不跟=楔住与否）。
     PC4/PC5 默认关闭（信息已内化）；PC2 精简到 2 行。

[v0.22 变更·依据 v0.21 首跑实测（组装顺序/停止点问题：过卷 + 收口时接触蒸发）]
  ① v0.21 实锤：
     a) 拇指滑入干净触到**稳定复现**（滑 14mm、Fth=0.177N、cubeΔ=0.2mm）✓；
     b) **手指过卷暴露**：j2/j3 实测能卷到 +1.40/+1.38（行程全真实），但最接近点在 ≈1.2
        （盒距 25mm）；再深时指尖从 z+3.6 升回 z+5.5——**从 cube 上方翻过去了**（无接触、cubeΔ 干净）
        → 手指必须停在"最接近点"，不能卷到底；
     c) **收口阶段两边接触全丢**：这次跑整手比上轮高 ~0.5cm（站位漂移），手指没碰到；拇指的
        0.177N 也在收口动作中"回缩"掉（指尖上退 1.4cm）→ 收口空转、微提无反应。
  ② v0.22（组装顺序重排 = [PC7]）：
     a) **先整体下放 3cm**（手指伸直/拇指开放、无接触+守卫）——把一切"从上方压顶棱"的接触
        变成"横向压面"（对提起方向才有意义）；
     b) 手指深卷带**过卷保护**（跟踪盒距：首次触到 Fmid>0.05 或盒距从 <50mm 回升即停）；
     c) 拇指合拢B + 滑入（Fth>0.10）；
     d) **收口只做小动作**：拇指沿滑入向 0.2mm 级续进（Fth>0.35/cubeΔ>1mm 停）→ 臂 +y 0.2mm 级微进
        （Fmax>0.60/cubeΔ>1.5mm 停）——全程监视"力是否保持"；
     e) 微提 +1mm 探测。PC6 默认关（教训已内化）。

[v0.23 变更·依据 v0.22 首跑实测（两侧"触到"都成立 → 主题转向"留住 + 对夹轴"）]
  ① v0.22 实锤：
     a) **降手 3cm 生效**——手指首次获得真接触（卷到 1.25 时 Fmid=0.519N 冲量）；但**"停指后力全消"**：
        冻前目标仍挂着 1.25，手指继续收敛→从棱上滚过去（到位 mid4 z=+0.5 → 末态 +3.6cm＝整段在后撤）。
     b) 拇指触到 0.310N 后，沿滑入向续进力**单调衰减**（0.237→0.004N）：滑入方向对已贴住的接触是"切向蹭走"。
     c) 臂 +y 时拇指断续有力（0.04~0.19N）、cubeΔ 0.1~0.2mm 干净，但手指各步为 0（当时指尖已滚到 z+3.6cm）。
     d) 本轮 cube 净位移 3.5mm（来自 +y 推）；微提无跟动。
  ② v0.23（触到即冻 · 对夹轴重测 = [PC7]）：
     a) 降手 3cm 保留；
     b) 手指深卷**触到即冻**：把 j2/j3/j4 目标改成"触到瞬间的实际角"（杜绝继续收敛滚走）→ 40 步**保力表**
        （每 5 步打印 Fmid，看"真保持"还是"松手即散"）；
     c) **手指侧对夹轴测试**：臂 +y 0.2mm×15 逐步行（Fmid 会不会随步升）——用整臂平移把冻住的手指压向 cube；
     d) 拇指合拢B+滑入（同 v0.22）**触到即冻** → 40 步拇指保力表（含对手指接触的交叉影响打印）；
     e) **拇指对夹轴探针**：-x×6 → 回位 → +y×8 → 回位 → 关节微捏 t3/t4 +0.04×6（找"哪个方向让 Fth 升"）；
     f) 末态判定 + 微提 +1mm。

[v0.24 变更·依据 v0.23 首跑实测（"冻结"解决滚走；力的本质=命令差 → 收口改用"微收/微捏"）]
  ① v0.23 实锤：
     a) 手指"触到即冻"后**位置稳住了**（不再滚走：盒距稳定 22~23mm、mid4 稳定在 z≈0）——但**力立刻归零**：
        PD 命令差=0 → 静态无压（0.417N 是动态冲量）；
     b) 手指臂 +y 推 2.6mm 才擦出 0.011~0.013N——"平移压"太慢；
     c) 拇指滑入冻住后 Fth 断续尖峰（0/0.179/0.449/0——擦边接触）；臂 -x 无效、+y 起点 0.186 后衰减；
     d) **拇指关节微捏（探针C）：+0.04 级持续出 0.169→0.290N，cubeΔ 仅 0.4mm**——"关节继续收=持续压"✓
        （拇指的收合方向正好指向接触；与手指"收就滚"不同）。
  ② v0.24（力的来源 = 命令差 → 收口用"微收/微捏"阶梯）：
     c) **手指加力轴**：冻结后 j2/j3 阶梯微收 +0.04×8（监控 Fmid/盒距；盒距回升+指尖上飘+无力=滚走→回退一级保留）；
     e) **拇指收口**：t3/t4 阶梯微捏 +0.04×8（Fth>0.35 或 cubeΔ>1.5mm 停）→ **不回退**，直接进复核；
     f) 微提 +1mm 复核。保力表缩到 20 步。

[v0.25 变更·依据 v0.24 首跑实测（★手指"微收=加力"成立；拇指滑入变脆 + 守卫被陈旧位移误杀）]
  ① v0.24 实锤：
     a) ★★ **手指微收=加力轴成立**：冻结后 +0.04 级微收 → Fmid 0→0.235N（收 0.08、j2实=1.16），
        持续 0.13~0.22N 到 j2实≈1.29，j2实≥1.34 后滚走——**甜点=收 0.04~0.24（j2实 1.15~1.29）**，
        且全程 **cubeΔ=0.0mm**（最干净的持续加力）；
     b) 拇指滑入这次变脆：滑入起步就蹭推 cube 1mm（无 Fth 读数）→ 停滑 → 滑入没触到（上轮 8mm 干净触到）；
     c) e) 微捏被"陈旧位移"守卫误杀（滑入残留 1mm 不清理，第一个捏步就报 3.8mm 停）；（末态 Fth=0.270
        说明拇指其实碰到了；f) 微提 cube_z +0.81mm 但力读数 0 = 蹭动/回弹嫌疑）。
  ② v0.25（收口定稿的三个修正）：
     c) 手指微收改**力停**：Fmid>0.15N 即停（停在甜点）；力过峰值后跌零→回退一级；
     d) 拇指滑入改 **0.5mm/循环慢滑**（每 8 循环/触到必打）+ 守卫收紧 0.8mm（带几何打印）；
     e) 拇指收口用**段独立基线**（防滑入残留误触）；力停 0.30N；d2 保力表加 cubeΔ；
     f) 微提判定区分"跟动+有力"与"蹭动无力"。

[v0.26 变更·依据 v0.25 首跑实测（拇指微捏收口 0→0.305N/cubeΔ≈0 成立；手指窄窗+漂移是最后一块）]
  ① v0.25 实锤：
     a) **拇指微捏收口成立**：滑 5.5mm 干净触到（Fth=0.083、cubeΔ=0.0）→ 冻结后 8 级 +0.04 微捏
        Fth 0.036→0.106→0.163→0.190→0.189→0.195→0.258→0.305 平稳爬升、**cubeΔ(段)=0.0mm**；
        停后自然收敛又爬到 **0.888N**（cubeΔ 总 0.6mm）——拇指侧已是可靠加力轴；
     b) 手指窗口**窄且游移**：本轮冻在 j2实=1.16（上轮 1.13）→微收窗口整体前移（收 0.04/0.08 只 0.07~0.08N、
        0.12 就跌零）——**冻结位 ±0.03 rad 就能让窗口移位**；c 段后手指无力，d/e 相位期间指尖继续
        缓慢收敛滚动、漂到 (+5.8,-3.5,+2.0)cm——完全脱离；
     c) 微提 cube_z -0.22mm、双力 0 —— 单侧拇指压（0.9N）不算夹持，**需双侧同框**。
  ② v0.26（双侧同框）：
     c)/e2) 新增手指例行 **finger_reach7**（退 0.90 → +0.05 逼近触到即冻 → -0.08 回撤 → +0.02 从下方扫，
        首达 Fmid>0.10 停保留；峰后跌零回峰位）——**c 先做（拇指前基线），e2 在拇指收口后再做一次**
        （验证拇指相位后能否重建手指接触 + 交叉影响）；
     e) 拇指微捏力停阈值 0.30→**0.15N**（后续自然收敛继续爬升），15 步保持打印；
     f) 微提仍 +1mm（双侧同框后再判楔住）。

[v0.27 变更·依据 v0.26 首跑实测（c/e 两段全通；e2 溃于腕漂移 → 新增"腕回锁"）]
  ① v0.26 实锤：
     a) ★ **c 段 finger_reach7 全通**：回撤后 +0.02 从下方扫——+0.12→0.096、+0.14→0.108 → 停保留、
        保持 0.066N，全程 cubeΔ=0.0mm（"从下方找力"例行成立）；
     b) 拇指滑入 3.0mm 即触（Fth=0.083）、保力表 0.245/0.172/0.127（比历史"全零"实）；
        微捏到 0.164 停 → 15 步自然爬升 **0.479N**、cubeΔ(段)=0.0（拇指链完美）；
     c) **e2 全败——腕漂移铁证**：e2 逼近到 1.55 一次没触到，细扫盒距乱跳（16→37→28mm），
        末态 wrist_3_joint=+4.03 rad（站位 +3.17）＝腕转了 ~49°、TCP 偏 ~6cm——手指弧线整套错位；
        拇指 0.479N 也在漂移中丢光；d) 合拢B 仍把手指 0.066 顶掉（老现象）。
  ② v0.27（腕回锁）：
     a) 新增 **relock7**（位姿伺服拉回 wp7/wq0，预算 150 步，|e_p|<2mm 且 rotErr<1° 收敛；
        打印前后 Fth/Fmid 变化）；
     b) finger_reach7：退 0.90 分 4 档柔性 + **逼近前调 relock7**；逼近上限 1.55→1.45；
        未触到→直接报告（不再回撤乱扫推 cube）；
     c) e 段前加 relock7("e-pre")；末态打印腕位姿诊断（>5mm/3° 判本轮存疑）。

[v0.28 变更·依据 v0.27 首跑实测（★腕回锁暴露真根因：**腕转动漂移 12~28° 且回锁方向被拧歪**）]
  ① v0.27 实锤：
     a) relock7 每次都"未收敛"：c 后 12.5°、e 前 28.3°、e2 前 24.2°、末态 24.3°——**位置误差只 1~3mm，
        但腕转动整段漂 12~28°**（手指快动/接触反力矩把腕"拧"走；位置伺服看不住转动）；
     b) 根因：hold_wrist_step 的转动修正是**逐分量 clamp 到 ±0.03 再 /0.005**——大误差时把修正方向
        拧歪（主轴被压平），腕在错误轴上打转、几乎不收敛→每个相位累积；
     c) 漂移把一切毁掉：c 的力窗找到 0.275@+0.10 但停后归零（"力被腕扭转泄掉"）、拇指滑入/收口值变弱、
        e2 逼近整个落空（盒距 43mm）。
  ② v0.28（转动方向保真 + 回锁强化）：
     a) hold_wrist_step / _servo_step 的转动命令改"沿误差轴缩放"（幅值限 0.03，再整体归一化到
        raw 上限）——方向不再被拧歪；
     b) relock7：预算 150→250、门限 rotErr<1°→<2°、每 50 步打进度；
     c) 新增 d-pre 回锁（拇指相位前清残差）；末态诊断保留。

跑法（远程 conda isaacsim5.1）:
    python scripts/scripted_grasp.py --task Template-Ur5e-Drillgrasp-v0 --num_envs 1

输出解读（发回控制台全文即可）：
    · [PC7] 表全看：
      -c/e2 回锁行：rotErr 能否收到 <2°（v0.27 是 12~28° 卡住）；
      -c 细扫：Fmid 停后能不能保持（转动修好后"力被拧泄"应改善）；
      -d 拇指慢滑 + -e 微捏：滑几 mm 触到、Fth 爬到多少；
      -f 微提 + 末态腕诊断；
    · [PH]/GUI：合围时两指各压 cube 的哪个部位（面/棱）。
"""

import argparse
import csv
import os
import time

from isaaclab.app import AppLauncher

# ── argparse ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Scripted grasp chain v0.2: arm cal + fast approach servo + in-situ hand cal + report.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Ur5e-Drillgrasp-v0", help="Name of the task.")
parser.add_argument("--settle_steps", type=int, default=10, help="reset 后的零动作短稳步数。")
parser.add_argument("--clearance", type=float, default=0.015, help="（旧参数，v0.13 起 PB 改用 TCP 对中，不再使用）")
parser.add_argument("--stop_z_rel", type=float, default=0.055, help="（旧参数，v0.13 起 PB' 改用首触停，不再使用）")
parser.add_argument("--step_clip", type=float, default=0.010, help="伺服每步最大位移（m）[v0.2: 4mm→10mm=raw2.0 上限]。")
parser.add_argument("--rot_hold", type=int, default=1, help="1=姿态保持（对抗腕漂移）；0=关。")
parser.add_argument("--hold_end", type=int, default=60, help="保持步数（供 GUI 观察）。")
parser.add_argument("--max_steps", type=int, default=700, help="接近伺服最大步数 [v0.2: 260→700]。")
parser.add_argument("--hand_cal", type=int, default=1, help="1=到位后做手部原地标定（PC）。")
parser.add_argument("--rehearsal", type=int, default=1, help="1=PC3 合拢预演（卷曲/按压/拇指，力限幅）。")
parser.add_argument("--press_force", type=float, default=0.08, help="PC3 中指触面力阈值（传感器若醒，N）。")
parser.add_argument("--squeeze_force", type=float, default=0.30, help="PC3 双侧压紧成立的早停阈值（拇指侧力，N）。")
parser.add_argument("--squeeze_top", type=float, default=0.90, help="v0.11 PC3 起飞前补压目标（拇指侧力，N）[v0.10: 0.45]。")
parser.add_argument("--thumb_touch", type=float, default=0.10, help="v0.7 PC3 拇指贴面力停（N）。")
parser.add_argument("--stage_x", type=float, default=0.0, help="v0.13 PC3 臂 -x 预置（m）[默认已改 0.0：正上方罩下时不需让角]。")
parser.add_argument("--hand_friction", type=float, default=2.0, help="v0.12 机器人碰撞材质摩擦（μ，0=不改动；绑定到全部碰撞 prim）。")
parser.add_argument("--engage_y", type=int, default=0, help="v0.15 [PZ+1]：四指簇慢速对准到 -y 面外（默认 0=关；会把整手推 +y ~6cm——引入 y 偏移，用户不要）。")
parser.add_argument("--mouth_align", type=int, default=0, help="v0.17 [B2] 口心对位接近（默认 0=关——姿势未定先不用；1=开）。")
parser.add_argument("--press", type=int, default=0, help="v0.17 加压/微提段总开关（默认 0=关——只观察姿势；1=跑 [5A][5B] 与微提）。")
parser.add_argument("--hand_clip", type=float, default=1.6, help="v0.18 手部动作 raw 限幅（默认 1.6：放宽以命令更深弯；仅改本脚本的 env_cfg，训练代码不动）。")
parser.add_argument("--pc4", type=int, default=0, help="v0.21 默认关（滑移信息已内化到 [PC6]；1=跑旧 D/合拢B 滑移）。")
parser.add_argument("--pc5", type=int, default=0, help="v0.21 默认关（手指可达性已确认，见 [PC6-1]；1=跑旧深卷预试）。")
parser.add_argument("--pc6", type=int, default=0, help="v0.22 默认关（教训已内化到 [PC7]；1=跑旧版合围）。")
parser.add_argument("--pc7", type=int, default=1, help="v0.28 [PC7] 双侧同框·转动伺服修正：降手→深卷·冻→手指例行(c)→回锁→拇指慢滑·冻→微捏→手指例行(e2)→微提（1=开）。")
parser.add_argument("--pc7_drop", type=float, default=0.030, help="[PC7-a] 组装前整体下放量（m；默认 30mm——把'压顶棱'变'横压面'）。")
parser.add_argument("--rehearsal_x", type=float, default=-0.030, help="（旧参数，v0.7 不再使用）")
parser.add_argument("--cal_amp", type=float, default=0.25, help="PC 每通道 nudge 幅值（raw）。")
parser.add_argument("--cal_hold", type=int, default=8, help="PC 每通道按住步数。")
parser.add_argument("--cal_rest", type=int, default=12, help="PC 通道间/前后静置步数。")
parser.add_argument("--out_dir", type=str, default="scripted_grasp_out", help="CSV 输出目录。")
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
from isaaclab.utils.math import quat_apply, quat_apply_inverse

import ur5e_drillgrasp.tasks  # noqa: F401
from ur5e_drillgrasp.tasks.manager_based.ur5e_drillgrasp.mdp import observations as mdp_obs


# ── 四元数小工具（wxyz）─────────────────────────────────────────────────
def _qmul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def _qconj(a):
    w, x, y, z = a.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def _rotvec(q):
    """四元数 → 旋转向量（rad，世界系）。"""
    if q[0] < 0:
        q = -q
    w = q[0].clamp(-1.0, 1.0)
    ang = 2.0 * torch.acos(w)
    if ang.item() < 1e-7:
        return torch.zeros(3, device=q.device)
    s = torch.sqrt(torch.clamp(1.0 - w * w, min=1e-12))
    return q[1:] / s * ang


def main():
    """v0.28：臂标定 + PB' + [PZ] + PC2 复现 + [PC7] 双侧同框·转动伺服修正（降手→深卷·冻→手指例行(c)→回锁→拇指慢滑·冻→微捏→手指例行(e2)→微提）。"""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.episode_length_s = 600.0  # 防中途重置
    try:
        env_cfg.actions.hand_action.clip_range = float(args_cli.hand_clip)  # v0.18: 运行时放宽手部 raw 限幅（仅本脚本）
    except Exception as _e:
        print(f"[v0.18] 设置 hand clip_range 失败（沿用默认 1.0）: {_e}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    uw = env.unwrapped
    dev = uw.device

    robot = uw.scene["robot"]
    cube = uw.scene["cube_obj"]
    joint_names = list(robot.joint_names)

    tip_names = list(mdp_obs._FINGERTIP_NAMES)
    body_id = {}
    for nm in tip_names:
        ids, _ = robot.find_bodies(nm)
        if len(ids) == 0:
            raise RuntimeError(f"找不到指尖 body: {nm}")
        body_id[nm] = int(ids[0])
    wrist_ids, _ = robot.find_bodies("wrist_3_link")
    wrist_id = int(wrist_ids[0])

    # ═════════════════ [MAT] v0.12：物理材质探查 + 手部高摩擦材质绑定 ═════════════════
    try:
        import omni.usd
        from pxr import Usd, UsdShade, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        scan_keys = ("thumb4", "middle4", "wrist_3")
        found = []
        for _p in Usd.PrimRange(stage.GetPseudoRoot()):
            _nm = _p.GetName()
            if any(k in _nm for k in scan_keys) or "cube" in _nm:
                found.append(_p)
        print("\n[MAT] 物理材质探查（绑定前）:")
        for _p in found:
            _bmr = UsdShade.MaterialBindingAPI(_p).ComputeBoundMaterial()
            _bm = _bmr[0] if isinstance(_bmr, (tuple, list)) else _bmr  # v0.14: 兼容 2/3 元组返回
            _sf = _dyn = None
            if _bm:
                _a = _bm.GetPrim().GetAttribute("physics:staticFriction")
                _b = _bm.GetPrim().GetAttribute("physics:dynamicFriction")
                _sf = _a.Get() if _a and _a.HasAuthoredValue() else None
                _dyn = _b.Get() if _b and _b.HasAuthoredValue() else None
            _a2 = _p.GetAttribute("physics:staticFriction")
            _sf2 = _a2.Get() if _a2 and _a2.HasAuthoredValue() else None
            print(f"     {_p.GetName():18s}: 材质={str(_bm.GetPath()) if _bm else None}  material_sf={_sf} df={_dyn}  prim_sf={_sf2}")
    except Exception as e:
        print(f"\n[MAT] 材质探查失败（不影响流程）: {e}")

    if args_cli.hand_friction > 0:
        try:
            import omni.usd
            from pxr import Usd, UsdShade, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            _mu = float(args_cli.hand_friction)
            _mpath = "/World/hand_hi_friction_mat"
            _mat = UsdShade.Material.Define(stage, _mpath)
            _mapi = UsdPhysics.MaterialAPI.Apply(_mat.GetPrim())
            _mapi.CreateStaticFrictionAttr(_mu)
            _mapi.CreateDynamicFrictionAttr(_mu)
            _mapi.CreateRestitutionAttr(0.0)
            try:
                from omni.physx.scripts import physicsUtils
                _use_pu = True
            except Exception:
                _use_pu = False
            _n_bind = 0
            for _p in Usd.PrimRange(stage.GetPseudoRoot()):
                _sp = str(_p.GetPath())
                if "/Robot" not in _sp:
                    continue
                if _p.HasAPI(UsdPhysics.CollisionAPI) or _p.HasAPI(UsdPhysics.MeshCollisionAPI):
                    if _use_pu:
                        physicsUtils.add_physics_material_to_prim(stage, _p, _mpath)
                    else:
                        UsdShade.MaterialBindingAPI.Apply(_p).Bind(_mat, UsdShade.Tokens.weakerThanDescendant)
                    _n_bind += 1
            print(f"[MAT] 手部高摩擦材质已绑定: {_mpath}（μ={_mu:.2f}）→ {_n_bind} 个碰撞 prim")
        except Exception as e:
            print(f"[MAT] 高摩擦材质绑定失败（继续用原材质）: {e}")

    # ── 快照 / 步进 ─────────────────────────────────────────────────────
    def snap():
        with torch.inference_mode():
            tips = {nm: robot.data.body_link_pos_w[0, body_id[nm]].clone() for nm in tip_names}
            return dict(
                tips=tips,
                wrist_p=robot.data.body_link_pos_w[0, wrist_id].clone(),
                wrist_q=robot.data.body_link_quat_w[0, wrist_id].clone(),
                jpos=robot.data.joint_pos[0].clone(),
                jvel=robot.data.joint_vel[0].clone(),
                forces=mdp_obs.fingertip_contact_force(uw)[0].clone(),
                cube_p=cube.data.root_pos_w[0].clone(),
                cube_q=cube.data.root_quat_w[0].clone(),
                tcp=mdp_obs.tcp_position(uw)[0].clone(),
            )

    actions = torch.zeros(env.action_space.shape, device=dev)

    def step_n(n: int):
        for _ in range(int(n)):
            with torch.inference_mode():
                env.step(actions)

    def tip_mean_of(s):
        return torch.stack([s["tips"][nm] for nm in tip_names]).mean(0)

    # ── 符号表（PA 更新；提前定义供 hold_wrist_step 使用）──
    s_pos = [1.0, 1.0, 1.0]
    s_rot = [1.0, 1.0, 1.0]

    def hold_wrist_step(wp0, wq0, hand_cmd=None):
        """一步：腕位姿保持（位置→wp0、姿态→wq0 的伺服）+ 可选手部 raw。
        v0.3 升为共用件：PA 休息段 / PB-A 归位 / PC / PC2 / PC3 全用它——
        零动作=顺从模式会自由滚（v0.2 教训：PA 期间攒 38°，下降末段突然收敛扫动 cube）。"""
        s = snap()
        e_wp = wp0 - s["wrist_p"]
        e_wr = _rotvec(_qmul(wq0, _qconj(s["wrist_q"])))
        raw = torch.zeros(6, device=dev)
        for d in range(3):
            raw[d] = s_pos[d] * float(torch.clamp(e_wp[d] / 0.005, -2.0, 2.0))
        if args_cli.rot_hold and e_wr.norm().item() > 0.010:  # v0.15: 死区 0.86°→0.57°，权限 1.0→2.0（PB-A 要能拉回 15° 残差）
            # v0.28 修正：v0.26/27 实测大 rotErr（12~28°）不回锁的根因——分量各自 clamp 会把修正方向拧歪；
            # 改为"沿误差轴缩放"：先按总幅值限 0.03，再整体归一化到 raw 最大分量=2.0（方向不变）
            en_wr = e_wr.norm().item()
            rc = e_wr * (min(0.03, en_wr) / en_wr) / 0.005
            m_rc = float(rc.abs().max())
            if m_rc > 2.0:
                rc = rc * (2.0 / m_rc)
            for d in range(3):
                raw[3 + d] = s_rot[d] * float(rc[d])
        actions.zero_()
        actions[0, :6] = raw
        if hand_cmd is not None:
            actions[0, 6:14] = hand_cmd
        with torch.inference_mode():
            env.step(actions)

    # ═════════════════ P0：零动作基线（v0.15: + 主动稳姿再锁 q_hold）═════════════════
    print("\n[P0] 零动作基线：reset + 短稳 + 主动稳姿")
    env.reset()
    step_n(args_cli.settle_steps)                # v0.2: 先让 reset 瞬态沉下去
    s0 = snap()
    wp_stab, wq_stab = s0["wrist_p"].clone(), s0["wrist_q"].clone()
    for _ in range(40):                          # v0.15: 40 步"保持当前位姿"——把宽容稳定残差压掉再取参考
        hold_wrist_step(wp_stab, wq_stab)
    s0 = snap()
    q_hold = s0["wrist_q"].clone()               # 姿态保持目标 = 主动稳定后的腕姿态（运行间可复现性更好）
    cube0 = s0["cube_p"].clone()                 # （v0.1 取在 reset+3 步：瞬态姿态被锁成目标）
    cube0_q = s0["cube_q"].clone()               # v0.18: cube 起始姿态（PC2 逐行/PC3 复位用）
    tm0 = tip_mean_of(s0)
    half = float(mdp_obs._CUBE_HALF_SIZE)
    print(f"     cube0=({cube0[0]:+.3f},{cube0[1]:+.3f},{cube0[2]:+.3f})  wrist0=({s0['wrist_p'][0]:+.3f},{s0['wrist_p'][1]:+.3f},{s0['wrist_p'][2]:+.3f})")
    print(f"     指尖均值0=({tm0[0]:+.3f},{tm0[1]:+.3f},{tm0[2]:+.3f})  jvel max={s0['jvel'].abs().max().item():.3f}")

    def restore_cube():
        """v0.18：把 cube 复位到起始位姿（PC2 逐行之间 + PC3 前用，防上一行的推挤污染下一行几何）。
        v0.19 修复：v0.18 首跑报 'Inplace update to inference tensor outside InferenceMode'——
        仿真缓冲在 inference 语境里创建过 → 写入必须同样包进 inference_mode（否则整个复位失效）。"""
        try:
            _ids = torch.tensor([0], device=dev)
            _pose = torch.cat([cube0, cube0_q]).unsqueeze(0)
            with torch.inference_mode():
                cube.write_root_pose_to_sim(_pose, env_ids=_ids)
                cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev), env_ids=_ids)
        except Exception as _e:
            print(f"     [restore_cube] 失败: {_e}")

    # ═════════════════ PA：臂自标定（符号表）═════════════════
    print("\n[PA] 臂自标定：逐维 nudge（测方向+符号；验证 raw×0.005 语义）")
    ex_axes = [torch.tensor(a, device=dev, dtype=torch.float32) for a in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]
    cal_plan = [(d, 0.8, 6, 12) for d in range(3)] + [(d, 1.5, 10, 12) for d in range(3, 6)]
    wp_pre = s0["wrist_p"].clone()
    for d, amp, hold_n, rest_n in cal_plan:
        for _ in range(rest_n):                  # v0.3: 休息段=腕位姿保持（PA 期间不再攒腕滚）
            hold_wrist_step(wp_pre, q_hold)
        sb = snap()
        actions.zero_()
        actions[0, d] = amp
        step_n(hold_n)
        sa = snap()
        dp = sa["wrist_p"] - sb["wrist_p"]
        dr = _rotvec(_qmul(sa["wrist_q"], _qconj(sb["wrist_q"])))
        if d < 3:
            dot = float(torch.dot(dp, ex_axes[d].to(dp.dtype)))
            s_pos[d] = 1.0 if dot > 0 else -1.0
            warn = "" if abs(dot) > 0.3 * dp.norm().item() else "  ⚠️轴向不符预期"
            print(f"     dim{d} +{amp}: Δp=({dp[0]*1000:+.1f},{dp[1]*1000:+.1f},{dp[2]*1000:+.1f})mm"
                  f" |Δ|={dp.norm().item()*1000:.1f}mm sign={s_pos[d]:+.0f}{warn}")
        else:
            dot = float(torch.dot(dr, ex_axes[d - 3].to(dr.dtype)))
            s_rot[d - 3] = 1.0 if dot > 0 else -1.0
            warn = "" if abs(dot) > 0.3 * dr.norm().item() else "  ⚠️轴向不符预期"
            print(f"     dim{d} +{amp}: Δrot=({torch.rad2deg(dr[0]):+.2f},{torch.rad2deg(dr[1]):+.2f},{torch.rad2deg(dr[2]):+.2f})°"
                  f" sign={s_rot[d - 3]:+.0f}{warn}")
        # v0.2: 反向 nudge 抵消（零动作=顺从模式不会自动回位）; v0.3: 休息段改腕位姿保持
        actions.zero_()
        actions[0, d] = -amp
        step_n(hold_n)
        for _ in range(rest_n):
            hold_wrist_step(wp_pre, q_hold)
    print(f"     → 符号表 s_pos={s_pos} s_rot={s_rot}（伺服按此换算 raw = s * Δ/0.005）")

    # ═════════════════ PB'：TCP 对中 + 垂直下降至首触（v0.13 用户指令：手掌必须落在 cube 正上方）═════════════════
    tcp_xy_t = cube0[:2].clone()                     # 目标：TCP 的 xy ↔ cube 质心 xy
    print(f"\n[PB'] TCP 对中：目标 xy=({tcp_xy_t[0]:+.3f},{tcp_xy_t[1]:+.3f})（cube 质心正上方）")

    # ── PB-A：姿态归位（位置保持在高位，先把 q_hold 姿态误差压下去再靠近）──
    # v0.2 教训：PA 期间累计 38° 腕滚，下降末段"腕滚突然收敛"的扫掠把 cube 推了 4.3mm。
    wp_pb = snap()["wrist_p"].clone()
    print("[PB-A] 姿态归位（安全高度先把 rotErr 压到 <3°；v0.15: 预算 300→800、权限恢复）")
    rr_end = None
    for it in range(800):
        s = snap()
        e_rot = _rotvec(_qmul(q_hold, _qconj(s["wrist_q"])))
        rr = torch.rad2deg(e_rot.norm()).item()
        rr_end = rr
        if rr < 3.0:
            print(f"     ✓ rotErr={rr:.1f}° @ {it} 步")
            break
        hold_wrist_step(wp_pb, q_hold)
        if it % 50 == 0:
            print(f"     it{it:3d} rotErr={rr:.1f}°")
    if rr_end is not None and rr_end >= 3.0:
        print(f"     ⚠️ rotErr 未达标（{rr_end:.1f}°）——姿态残余，下降将带倾斜（记入日志）")

    def _servo_step(target_wp, max_delta=0.008):
        """一步：腕 → target_wp（世界系 P 伺服，逐步钳幅）+ 姿态保持 q_hold。"""
        s = snap()
        e_wp = target_wp - s["wrist_p"]
        e_wr = _rotvec(_qmul(q_hold, _qconj(s["wrist_q"])))
        cmd = torch.clamp(e_wp, -max_delta, max_delta)
        raw = torch.zeros(6, device=dev)
        for d in range(3):
            raw[d] = s_pos[d] * float(torch.clamp(cmd[d] / 0.005, -2.0, 2.0))
        if args_cli.rot_hold and e_wr.norm().item() > 0.010:  # v0.15: 死区 0.57°（下降段保持保守钳幅 1.0）；v0.28: 方向保真
            en_sr = e_wr.norm().item()
            rc_s = e_wr * (min(0.03, en_sr) / en_sr) / 0.005
            m_s = float(rc_s.abs().max())
            if m_s > 1.0:
                rc_s = rc_s * (1.0 / m_s)
            for d in range(3):
                raw[3 + d] = s_rot[d] * float(rc_s[d])
        actions.zero_()
        actions[0, 0:6] = raw
        with torch.inference_mode():
            env.step(actions)

    # 阶段 1：xy 对中（高度保持；伺服 TCP 的 xy → cube 质心 xy）
    for it in range(500):
        s = snap()
        e_xy = tcp_xy_t - s["tcp"][:2]
        en = e_xy.norm().item()
        if it % 20 == 0:
            rr = torch.rad2deg(_rotvec(_qmul(q_hold, _qconj(s["wrist_q"]))).norm()).item()
            print(f"     [PB'-xy] it{it:3d} TCP=({s['tcp'][0]:+.3f},{s['tcp'][1]:+.3f},{s['tcp'][2]:+.3f}) |e_xy|={en*1000:5.1f}mm rotErr={rr:.1f}°")
        if en < 0.004:
            print(f"     ✓ xy 对中完成 |e_xy|={en*1000:.1f}mm @ {it} 步")
            break
        tgt = s["wrist_p"].clone()
        tgt[0] += float(torch.clamp(e_xy[0], -0.010, 0.010))
        tgt[1] += float(torch.clamp(e_xy[1], -0.010, 0.010))
        _servo_step(tgt)

    # 阶段 2：垂直下降（v0.15：两段速 + xy 主动保持；停 = **TCP 到达 cube 质心高度**——用户指令"TCP 刚好在质心"）
    z_slow = float(cube0[2]) + 0.055          # 快转慢门限（TCP 高于质心 5.5cm 后转慢速）
    touch = False
    for it in range(900):
        s = snap()
        f_th = s["forces"][0].item()
        f_mid = s["forces"][2].item()
        f_max = float(s["forces"].max())
        e_xy = tcp_xy_t - s["tcp"][:2]
        dz_tcp = float(s["tcp"][2]) - float(cube0[2])
        tip_z = min(float(s["tips"][nm][2]) for nm in tip_names)
        if it % 10 == 0:
            print(f"     [PB'-z] it{it:3d} TCP=({s['tcp'][0]:+.3f},{s['tcp'][1]:+.3f},{s['tcp'][2]:.4f})"
                  f"  dzTCP={dz_tcp*1000:+6.1f}mm |e_xy|={e_xy.norm().item()*1000:4.1f}mm"
                  f"  minTipZ=顶{(tip_z-cube0[2]-half)*1000:+5.1f}mm  Fth={f_th:.3f} Fmid={f_mid:.3f}N")
        if f_max > 0.05:
            touch = True
            who = "/".join(nm for k, nm in enumerate(tip_names) if float(s["forces"][k]) > 0.05)
            print(f"     ✓ 首触（{who}；Fmax={f_max:.3f}N）TCP_z={s['tcp'][2]:.4f} @ {it} 步"
                  f"  五分力=" + " ".join(f"{float(v):.3f}" for v in s["forces"]))
            break
        if dz_tcp <= 0.0:
            print(f"     ✓ TCP 到位：dzTCP={dz_tcp*1000:+.1f}mm（TCP≈cube 质心）@ {it} 步"
                  "（未接触——正常：干净站位，指尖几何见 [PZ]）")
            break
        if (s["cube_p"] - cube0).norm().item() > 0.002:
            print("     ⚠️ cube 被推动 → 停下降")
            break
        lead = 0.009 if dz_tcp > 0.055 else 0.0025   # 快段 ≈1.0mm/步；慢段 ≈0.33mm/步
        tgt = s["wrist_p"].clone()
        tgt[0] += float(torch.clamp(e_xy[0], -0.006, 0.006))   # v0.14: xy 主动保持（防 v0.13 的 +x 侧滑）
        tgt[1] += float(torch.clamp(e_xy[1], -0.006, 0.006))
        tgt[2] -= lead
        _servo_step(tgt)
    s = snap()
    wp_stage = s["wrist_p"].clone()
    print(f"     [PB'] 站位锁定: wrist=({wp_stage[0]:+.3f},{wp_stage[1]:+.3f},{wp_stage[2]:+.3f})"
          f"  TCP_rel_cube=({(s['tcp'][0]-cube0[0])*1000:+.0f},{(s['tcp'][1]-cube0[1])*1000:+.0f},{(s['tcp'][2]-cube0[2])*1000:+.0f})mm")

    # ═════════ [PZ] v0.14：站位诊断（五指/wrist 的 cube 局部坐标 + 簇心/口心）═════════
    s = snap()
    cq_z, cp_z = s["cube_q"].clone(), s["cube_p"].clone()

    def to_cube_z(p):
        return quat_apply_inverse(cq_z.unsqueeze(0), (p - cp_z).unsqueeze(0))[0]

    rel_pts = {nm: to_cube_z(s["tips"][nm]) for nm in tip_names}
    print("     [PZ] 站位诊断（cube 局部系 cm；立方体面在 ±2.5）：")
    for nm in tip_names:
        r = rel_pts[nm]
        print(f"        {nm:8s} ({r[0]*100:+.1f},{r[1]*100:+.1f},{r[2]*100:+.1f})")
    rwz = to_cube_z(s["wrist_p"])
    print(f"        wrist    ({rwz[0]*100:+.1f},{rwz[1]*100:+.1f},{rwz[2]*100:+.1f})"
          f"  TCP_rel=({(s['tcp'][0]-cp_z[0])*1000:+.0f},{(s['tcp'][1]-cp_z[1])*1000:+.0f},{(s['tcp'][2]-cp_z[2])*1000:+.0f})mm")
    c4 = torch.stack([rel_pts[nm] for nm in ("index4", "middle4", "ring4", "little4")]).mean(0)
    mouth = 0.5 * (rel_pts["thumb4"] + rel_pts["middle4"])
    mspan = (rel_pts["thumb4"] - rel_pts["middle4"]).norm()
    print(f"     四指簇心=({c4[0]*100:+.1f},{c4[1]*100:+.1f},{c4[2]*100:+.1f})"
          f"  口心(拇-中)=({mouth[0]*100:+.1f},{mouth[1]*100:+.1f},{mouth[2]*100:+.1f})  口距={mspan*100:.1f}cm")

    # ═════════ [PZ+1] v0.14：四指簇对准（慢速滑动 → 簇心到 -y 面外 1.6cm；全守卫）═════════
    engage_done = False
    if args_cli.engage_y and max(float(v) for v in s["forces"]) < 0.03:
        cube_refE = s["cube_p"].clone()
        tgt_c = torch.tensor([0.0, -(half + 0.016), 0.0], device=dev)
        z_hold = float(s["wrist_p"][2])
        print(f"     [PZ+1] 四指簇对准：目标 = 簇心 → (0.0, -{(half+0.016)*100:.1f}, ·)（面外 1.6cm，含 12mm 表面标定余量）")
        for it in range(220):
            s = snap()
            t4 = torch.stack([to_cube_z(s["tips"][nm]) for nm in ("index4", "middle4", "ring4", "little4")]).mean(0)
            e_c = tgt_c[:2] - t4[:2]
            en = e_c.norm().item()
            f_max = float(s["forces"].max())
            dcb = (s["cube_p"] - cube_refE).norm().item()
            if it % 10 == 0:
                print(f"     [PZ+1] it{it:3d} 簇心=({t4[0]*100:+.1f},{t4[1]*100:+.1f},{t4[2]*100:+.1f})"
                      f" |e|={en*1000:4.1f}mm Fmax={f_max:.3f}N cubeΔ={dcb*1000:.1f}mm")
            if en < 0.003:
                engage_done = True
                print(f"     ✓ 簇心对准完成 |e|={en*1000:.1f}mm @ {it} 步")
                break
            if f_max > 0.03:
                engage_done = True
                print(f"     ✓ 对准中触到面（Fmax={f_max:.3f}N）@ {it} 步 → 停（保持当前接触）")
                break
            if dcb > 0.002:
                print(f"     ⚠️ cube 被推 {dcb*1000:.1f}mm → 停对准")
                break
            mv = torch.clamp(e_c, -0.004, 0.004)
            mv_w = quat_apply(s["cube_q"].unsqueeze(0), torch.stack([mv[0], mv[1], torch.zeros_like(mv[0])]).unsqueeze(0))[0]
            tgt = s["wrist_p"].clone()
            tgt[0] += float(mv_w[0])
            tgt[1] += float(mv_w[1])
            tgt[2] += float(torch.clamp(torch.tensor(z_hold, device=dev) - s["wrist_p"][2], -0.002, 0.002))
            _servo_step(tgt)
        s = snap()
        wp_stage = s["wrist_p"].clone()          # PH 保持位 = 对准后的站姿
    else:
        print("     [PZ+1] 跳过（v0.15 默认关闭 --engage_y=0；或已有接触）")

    # ═════════════════ PC：手部原地标定（in-situ，按压/绕-合规划依据）═════════════════
    cal_rows = []
    if args_cli.hand_cal:
        s_ready = snap()
        wp0 = s_ready["wrist_p"].clone()
        wq0 = s_ready["wrist_q"].clone()
        cq0, cp0 = s_ready["cube_q"].clone(), s_ready["cube_p"].clone()

        def to_cube0(p):
            return quat_apply_inverse(cq0.unsqueeze(0), (p - cp0).unsqueeze(0))[0]

        def step_hold_wrist(hand_cmd=None):
            """一步：腕位姿保持（到 PC 起点姿态）+ 指定手部 raw。"""
            hold_wrist_step(wp0, wq0, hand_cmd)

        ch_names = ["thumb1", "thumb2", "thumb3", "thumb4", "four_j1", "four_j2", "four_j3", "four_j4"]
        print(f"\n[PC] 手部原地标定（到位姿态，腕位姿锁住）：每通道 +{args_cli.cal_amp} raw × {args_cli.cal_hold} 步")
        print("     ΔW=腕局部系(cm)｜ΔC=cube 局部系(cm)。拇指通道看 thumb4；四指通道看 middle4（index/ring/little 同组）。")
        for ci, nm in enumerate(ch_names):
            for _ in range(args_cli.cal_rest):
                step_hold_wrist()
            sb = snap()
            qw_b = sb["wrist_q"].clone()
            hand_cmd = torch.zeros(8, device=dev)
            hand_cmd[ci] = args_cli.cal_amp
            for _ in range(args_cli.cal_hold):
                step_hold_wrist(hand_cmd)
            sa = snap()
            dW = {t: quat_apply_inverse(qw_b.unsqueeze(0), (sa["tips"][t] - sb["tips"][t]).unsqueeze(0))[0] for t in tip_names}
            dC = {t: to_cube0(sa["tips"][t]) - to_cube0(sb["tips"][t]) for t in tip_names}
            fmax = sa["forces"].max().item()
            dcube = (sa["cube_p"] - cube0).norm().item()
            tag = ""
            if fmax > 0.03 or dcube > 0.004:
                tag = f"  ⚠️接触/扰动(Fmax={fmax:.3f}N, cubeΔ={dcube*1000:.1f}mm)——本行不可信"
            fmt = lambda v: f"({v[0]*100:+.2f},{v[1]*100:+.2f},{v[2]*100:+.2f})"
            print(f"     {nm:8s}+{args_cli.cal_amp}: thumb4 ΔW={fmt(dW['thumb4'])} ΔC={fmt(dC['thumb4'])}"
                  f" | middle4 ΔW={fmt(dW['middle4'])} ΔC={fmt(dC['middle4'])}{tag}")
            print(f"                index4 ΔW={fmt(dW['index4'])}  ring4 ΔW={fmt(dW['ring4'])}  little4 ΔW={fmt(dW['little4'])}")
            cal_rows.append((nm, dW, dC))
            for _ in range(args_cli.cal_rest):
                step_hold_wrist()

        # 指腹法线（v0.3）：pad = link 局部 -z（用户 2026-09-19 GUI 确认四指）→ 换算到 cube 系
        print("     [指腹法线] link 局部 -z 在 cube 系的方向（四指 pad=局部 -z；拇指附全轴映射供判读）：")
        for nm2 in ("middle4", "thumb4"):
            ql = robot.data.body_link_quat_w[0, body_id[nm2]]
            nz = quat_apply_inverse(cq0.unsqueeze(0), quat_apply(ql.unsqueeze(0), torch.tensor([[0.0, 0.0, -1.0]], device=dev)))[0]
            ax = {}
            for lab, vv in (("x", (1.0, 0.0, 0.0)), ("y", (0.0, 1.0, 0.0)), ("z", (0.0, 0.0, 1.0))):
                wv = quat_apply(ql.unsqueeze(0), torch.tensor([vv], device=dev))[0]
                ax[lab] = quat_apply_inverse(cq0.unsqueeze(0), wv.unsqueeze(0))[0]
            print(f"        {nm2:8s} pad(-z)=({nz[0]:+.2f},{nz[1]:+.2f},{nz[2]:+.2f})"
                  f" |轴映射 x=({ax['x'][0]:+.2f},{ax['x'][1]:+.2f},{ax['x'][2]:+.2f})"
                  f" y=({ax['y'][0]:+.2f},{ax['y'][1]:+.2f},{ax['y'][2]:+.2f})"
                  f" z=({ax['z'][0]:+.2f},{ax['z'][1]:+.2f},{ax['z'][2]:+.2f})")

    # ═════════════════ PC2：拇指可达性预试（组合通道 → 指尖能否进入对侧交战区）═════════════════
    if args_cli.hand_cal:
        zone = torch.tensor([0.012, half + 0.008, 0.0], device=dev)  # v0.4: +x/+y 角区（拇指-中指夹持目标点）
        combos = [
            ("D复现 t1=1.5,t2=1,t3=1,t4=1.5", [1.5, 1.0, 1.0, 1.5]),
            ("合拢B 对照 t1=1,t2=1,t3=.8,t4=.8", [1.0, 1.0, 0.8, 0.8]),
        ]
        print("\n[PC2] 拇指姿势复现（v0.21：2 行核对；每行前复位 cube）")
        print("     看：①盒距（指尖→cube 表面，越小越近）②指腹朝向 pad（想要 -x 分量）③拇指四关节实际角度 j=")
        jidx = {nm: i for i, nm in enumerate(joint_names)}
        for name, vals in combos:
            restore_cube()                        # v0.18: 防上一行推挤污染本行几何
            for _ in range(args_cli.cal_rest):
                step_hold_wrist()
            hand_cmd = torch.zeros(8, device=dev)
            hand_cmd[:4] = torch.tensor(vals, device=dev)
            for _ in range(4 * args_cli.cal_hold):
                step_hold_wrist(hand_cmd)
            sa = snap()
            rel = to_cube0(sa["tips"]["thumb4"])
            dq = rel - torch.clamp(rel, -half, half)
            dbox = float(dq.norm()) * 1000.0
            fmax = sa["forces"].max().item()
            dcube = (sa["cube_p"] - cube0).norm().item()
            qtp = robot.data.body_link_quat_w[0, body_id["thumb4"]]
            npad = quat_apply_inverse(sa["cube_q"].unsqueeze(0), quat_apply(qtp.unsqueeze(0), torch.tensor([[0.0, 0.0, -1.0]], device=dev)))[0]
            jt = [sa["jpos"][jidx[f"thumb{k}_joint"]].item() for k in (1, 2, 3, 4)]
            jts = " ".join(f"{v:+.2f}" for v in jt)
            print(f"     {name:32s} thumb4=({rel[0]*100:+.1f},{rel[1]*100:+.1f},{rel[2]*100:+.1f})cm 盒距={dbox:4.0f}mm"
                  f" pad=({npad[0]:+.2f},{npad[1]:+.2f},{npad[2]:+.2f}) j=[{jts}] Fmax={fmax:.3f}N cubeΔ={dcube*1000:.1f}mm")
            for _ in range(args_cli.cal_rest):
                step_hold_wrist()

        # ═════════════════ [PC4] v0.19：深姿贴背滑移（手指定住、手整体滑向"背面目标点"）═════════════════
        #   v0.18 结论：深弯能把拇指摆到"右后上角外 ~17mm"（历史最近），但**闭合弧线本身到不了背面**——
        #   差的是"手整体挪 2~3cm"。本段对两个候选终姿各做一次：摆好姿势（手指从此不动）→
        #   沿"指尖→背面目标点"方向 1mm/循环（4 步）慢滑 → Fmax>0.10N（触到）或 cubeΔ>2mm（推走）停
        #   → 观察 40 步报接触几何。**直接回答"拇指能不能贴到对侧（背面）"。**
        if args_cli.pc4:
            for tag4, vals4, tgt4 in (
                ("D姿  (1.5,1,1,1.5)", [1.5, 1.0, 1.0, 1.5], (0.008, half + 0.012, 0.008)),
                ("合拢B (1,1,.8,.8)", [1.0, 1.0, 0.8, 0.8], (0.000, half + 0.012, 0.000)),
            ):
                restore_cube()
                for _ in range(args_cli.cal_rest):
                    step_hold_wrist()
                hand4 = torch.zeros(8, device=dev)
                hand4[:4] = torch.tensor(vals4, device=dev)
                for _ in range(40):
                    step_hold_wrist(hand4)
                s = snap()
                t_tip = to_cube0(s["tips"]["thumb4"])
                d4v = t_tip - torch.clamp(t_tip, -half, half)
                tgt = torch.tensor(tgt4, device=dev)
                dir_v = tgt - t_tip
                dist = float(dir_v.norm())
                print(f"\n     [PC4] {tag4} 就位: thumb4=({t_tip[0]*100:+.1f},{t_tip[1]*100:+.1f},{t_tip[2]*100:+.1f})cm 盒距={d4v.norm().item()*1e3:.0f}mm"
                      f" Fmax={float(s['forces'].max()):.3f}N → 滑向背面目标 ({tgt[0]*100:+.1f},{tgt[1]*100:+.1f},{tgt[2]*100:+.1f})cm 共 {dist*1000:.0f}mm（1mm/循环；Fmax>0.10N 或 cubeΔ>2mm 停）")
                cube_ref4 = s["cube_p"].clone()
                step_w = quat_apply(s["cube_q"].unsqueeze(0), (dir_v / max(dist, 1e-6) * 0.001).unsqueeze(0))[0]
                wp4 = s["wrist_p"].clone()
                n_sl = min(int(dist / 0.001) + 1, 45)
                touched4 = False
                for k in range(n_sl):
                    wp4 = wp4 + step_w
                    for _ in range(4):
                        hold_wrist_step(wp4, wq0, hand4)
                    s = snap()
                    fmax4 = float(s["forces"].max())
                    dcb4 = (s["cube_p"] - cube_ref4).norm().item()
                    if k % 2 == 1 or fmax4 > 0.02:
                        r4 = to_cube0(s["tips"]["thumb4"])
                        d4v = r4 - torch.clamp(r4, -half, half)
                        print(f"     [PC4] 滑{(k+1)*1.0:4.1f}mm: thumb4=({r4[0]*100:+.1f},{r4[1]*100:+.1f},{r4[2]*100:+.1f})cm 盒距={d4v.norm().item()*1e3:3.0f}mm"
                              f" Fth={s['forces'][0]:.3f} Fmax={fmax4:.3f}N cubeΔ={dcb4*1000:.1f}mm")
                    if fmax4 > 0.10:
                        touched4 = True
                        print(f"     [PC4] ✓ 触到（Fmax={fmax4:.3f}N）@ 滑 {(k+1)*1.0:.1f}mm → 停滑")
                        break
                    if dcb4 > 0.002:
                        print(f"     [PC4] ⚠️ cube 被推 {dcb4*1000:.1f}mm → 停滑（接触即推走）")
                        break
                # v0.20: 触后加压（0.25mm/循环）——"顶得住还是推得走"的直接测量
                if touched4:
                    print("     [PC4] 触后加压（0.25mm/循环；Fmax>0.40N 或 cubeΔ>1mm 停）…")
                    for kp in range(16):
                        wp4 = wp4 + step_w * 0.25
                        for _ in range(2):
                            hold_wrist_step(wp4, wq0, hand4)
                        s = snap()
                        fth_p = s["forces"][0].item()
                        fmax_p = float(s["forces"].max())
                        dcb_p = (s["cube_p"] - cube_ref4).norm().item()
                        if kp % 2 == 1 or fth_p > 0.15:
                            print(f"        加压{(kp+1)*0.25:4.2f}mm: Fth={fth_p:.3f} Fmax={fmax_p:.3f}N cubeΔ={dcb_p*1000:.1f}mm")
                        if fmax_p > 0.40:
                            print("        ★ 力到 0.40N 停（cubeΔ 仍小 = 顶住了）")
                            break
                        if dcb_p > 0.001:
                            print("        ⚠️ cube 被推 1mm 停（顶不住/推走）")
                            break
                print("     [PC4] 触后观察 40 步（GUI 请看：拇指前缘压到 cube 的哪个部位）…")
                for _ in range(40):
                    hold_wrist_step(wp4, wq0, hand4)
                s = snap()
                r4 = to_cube0(s["tips"]["thumb4"])
                d4v = r4 - torch.clamp(r4, -half, half)
                q4 = robot.data.body_link_quat_w[0, body_id["thumb4"]]
                n4 = quat_apply_inverse(s["cube_q"].unsqueeze(0), quat_apply(q4.unsqueeze(0), torch.tensor([[0.0, 0.0, -1.0]], device=dev)))[0]
                print(f"     [PC4] {tag4} 结果: thumb4=({r4[0]*100:+.1f},{r4[1]*100:+.1f},{r4[2]*100:+.1f})cm 盒距={d4v.norm().item()*1e3:.0f}mm"
                      f" pad=({n4[0]:+.2f},{n4[1]:+.2f},{n4[2]:+.2f}) Fmax={float(s['forces'].max()):.3f}N cubeΔ={(s['cube_p']-cube_ref4).norm().item()*1000:.1f}mm"
                      f"（盒距最小↔贴面；看 x/y/z 哪个分量还大=还差的进给方向）")
                for _ in range(30):                    # 保持深姿、手腕先撤（防松手扫到 cube）
                    step_hold_wrist(hand4)
                hand4.zero_()
                for _ in range(20):
                    step_hold_wrist(hand4)

        # ═════════════════ [PC5] v0.20：深卷四指 + 双侧合围预试（手指行程也解锁）═════════════════
        #   动机：四指此前一直只卷到 1.0（和拇指被 clip 限住同类问题）；本段 j2/j3 渐进深卷到 1.4 →
        #   ①量 mid4 到底能够多近（能否碰前面）②叠拇指（D→合拢B）看双侧同时到位的力 ③慢滑 −x 看能否夹住。
        if args_cli.pc5:
            restore_cube()
            for _ in range(args_cli.cal_rest):
                step_hold_wrist()
            hand5 = torch.zeros(8, device=dev)
            cube_ref5 = snap()["cube_p"].clone()
            print("\n     [PC5a] 深卷四指（j2/j3 → 1.4 解锁行程；cubeΔ>2mm 停）")
            for kk in range(1, 29):
                tc = 1.4 * kk / 28
                hand5[5], hand5[6], hand5[7] = tc, tc, 0.30
                for _ in range(2):
                    step_hold_wrist(hand5)
                s = snap()
                dcb5 = (s["cube_p"] - cube_ref5).norm().item()
                if kk % 6 == 0:
                    r5 = to_cube0(s["tips"]["middle4"])
                    d5 = r5 - torch.clamp(r5, -half, half)
                    print(f"        j2/j3→{tc:.2f}: mid4=({r5[0]*100:+.1f},{r5[1]*100:+.1f},{r5[2]*100:+.1f})cm 盒距={d5.norm().item()*1e3:3.0f}mm"
                          f" Fmid={s['forces'][2]:.3f}N cubeΔ={dcb5*1000:.1f}mm")
                if dcb5 > 0.002:
                    print("        ⚠️ cube 被推 → 停卷")
                    break
            for _ in range(20):
                step_hold_wrist(hand5)
            s = snap()
            r5 = to_cube0(s["tips"]["middle4"])
            d5 = r5 - torch.clamp(r5, -half, half)
            jm5 = [s["jpos"][jidx[f"middle{k}_joint"]].item() for k in (2, 3, 4)]
            jm5s = " ".join(f"{v:+.2f}" for v in jm5)
            print(f"     [PC5a] 末态: mid4=({r5[0]*100:+.1f},{r5[1]*100:+.1f},{r5[2]*100:+.1f})cm 盒距={d5.norm().item()*1e3:.0f}mm"
                  f" j_mid=[{jm5s}] Fmid={s['forces'][2]:.3f}N")
            for tag5, tv5 in (("D", [1.5, 1.0, 1.0, 1.5]), ("合拢B", [1.0, 1.0, 0.8, 0.8])):
                hand5[:4] = torch.tensor(tv5, device=dev)
                for _ in range(35):
                    step_hold_wrist(hand5)
                s = snap()
                rT = to_cube0(s["tips"]["thumb4"])
                rM = to_cube0(s["tips"]["middle4"])
                dT = rT - torch.clamp(rT, -half, half)
                dM = rM - torch.clamp(rM, -half, half)
                print(f"     [PC5b] 深卷+拇指{tag5}: thumb4=({rT[0]*100:+.1f},{rT[1]*100:+.1f},{rT[2]*100:+.1f})cm 盒距={dT.norm().item()*1e3:3.0f}mm |"
                      f" mid4=({rM[0]*100:+.1f},{rM[1]*100:+.1f},{rM[2]*100:+.1f})cm 盒距={dM.norm().item()*1e3:3.0f}mm"
                      f" Fth={s['forces'][0]:.3f} Fmid={s['forces'][2]:.3f}N cubeΔ={(s['cube_p']-cube_ref5).norm().item()*1000:.1f}mm")
            print("     [PC5c] 双侧在位，慢滑 −x（0.5mm/循环，至多 1.2cm；Fmax>0.40N 或 cubeΔ>1mm 停）…")
            dir_w5 = quat_apply(s["cube_q"].unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=dev))[0]
            wp5 = s["wrist_p"].clone()
            for ks in range(24):
                wp5 = wp5 - dir_w5 * 0.0005
                for _ in range(2):
                    hold_wrist_step(wp5, wq0, hand5)
                s = snap()
                fth5 = s["forces"][0].item()
                fmid5 = s["forces"][2].item()
                fmax5 = float(s["forces"].max())
                dcb5 = (s["cube_p"] - cube_ref5).norm().item()
                if ks % 4 == 3 or fmax5 > 0.12:
                    print(f"        滑{(ks+1)*0.5:4.1f}mm: Fth={fth5:.3f} Fmid={fmid5:.3f} Fmax={fmax5:.3f}N cubeΔ={dcb5*1000:.1f}mm")
                if fmax5 > 0.40:
                    print("        ✓ 力到 0.40N → 停滑")
                    break
                if dcb5 > 0.001:
                    print("        ⚠️ cube 被推 1mm → 停滑")
                    break
            for _ in range(40):
                step_hold_wrist(hand5)
            s = snap()
            rT = to_cube0(s["tips"]["thumb4"])
            rM = to_cube0(s["tips"]["middle4"])
            print(f"     [PC5c] 末态: thumb4=({rT[0]*100:+.1f},{rT[1]*100:+.1f},{rT[2]*100:+.1f})cm | mid4=({rM[0]*100:+.1f},{rM[1]*100:+.1f},{rM[2]*100:+.1f})cm"
                  f" Fth={s['forces'][0]:.3f} Fmid={s['forces'][2]:.3f}N cubeΔ={(s['cube_p']-cube_ref5).norm().item()*1000:.1f}mm"
                  f"（Fth/Fmid 双升且 cubeΔ 小 = 夹住）")
            hand5[:4] = 0.0
            for _ in range(20):
                step_hold_wrist(hand5)
            hand5.zero_()
            for _ in range(20):
                step_hold_wrist(hand5)

        # ═════════════════ [PC6] v0.21：首次组装"合围"（手指深卷 → 拇指滑入 → 收口楔住 → 微提探测）═════════════════
        #   依据 v0.20：①四指深卷 j2/j3→1.4 能干净碰到前面（Fmid=0.206N、cubeΔ=0）——手指可达性解决；
        #   ②拇指合拢B 滑入能干净触到右后角（Fth≈0.13-0.23N、cubeΔ=0）——拇指干净接触也有；③D 姿闭合会推
        #   cube（12.4mm）——组装序列禁用 D。本段把两边第一次装上：手指到位 → 拇指到位 → 收口 → 微提验证。
        if args_cli.pc6:
            restore_cube()
            for _ in range(args_cli.cal_rest):
                step_hold_wrist()
            hand6 = torch.zeros(8, device=dev)
            cube_ref6 = snap()["cube_p"].clone()
            print("\n     [PC6-1] 手指深卷（j2/j3 → 1.4；首触 Fmid>0.15 停；cubeΔ>2mm 停）")
            for kk in range(1, 29):
                tc = 1.4 * kk / 28
                hand6[5], hand6[6], hand6[7] = tc, tc, 0.30
                for _ in range(2):
                    step_hold_wrist(hand6)
                s = snap()
                fm6 = s["forces"][2].item()
                dcb6 = (s["cube_p"] - cube_ref6).norm().item()
                if kk % 6 == 0:
                    r6 = to_cube0(s["tips"]["middle4"])
                    d6 = r6 - torch.clamp(r6, -half, half)
                    print(f"        j2/j3→{tc:.2f}: mid4=({r6[0]*100:+.1f},{r6[1]*100:+.1f},{r6[2]*100:+.1f})cm 盒距={d6.norm().item()*1e3:3.0f}mm Fmid={fm6:.3f}N cubeΔ={dcb6*1000:.1f}mm")
                if fm6 > 0.15:
                    print(f"        ✓ 手指触到（Fmid={fm6:.3f}N @ j2/j3→{tc:.2f}）")
                    break
                if dcb6 > 0.002:
                    print("        ⚠️ cube 被推 → 停卷")
                    break
            for _ in range(15):
                step_hold_wrist(hand6)
            s = snap()
            r6 = to_cube0(s["tips"]["middle4"])
            d6 = r6 - torch.clamp(r6, -half, half)
            jm6 = [s["jpos"][jidx[f"middle{k}_joint"]].item() for k in (2, 3)]
            print(f"     [PC6-1] 到位: mid4=({r6[0]*100:+.1f},{r6[1]*100:+.1f},{r6[2]*100:+.1f})cm 盒距={d6.norm().item()*1e3:.0f}mm"
                  f" j_mid=[{jm6[0]:+.2f} {jm6[1]:+.2f}] Fmid={s['forces'][2]:.3f}N cubeΔ={(s['cube_p']-cube_ref6).norm().item()*1000:.1f}mm")
            # 2) 拇指合拢B（空中干净闭合）+ 滑入右后角
            hand6[:4] = torch.tensor([1.0, 1.0, 0.8, 0.8], device=dev)
            for _ in range(35):
                step_hold_wrist(hand6)
            s = snap()
            rT = to_cube0(s["tips"]["thumb4"])
            dT6 = rT - torch.clamp(rT, -half, half)
            tgt6 = torch.tensor([0.000, half + 0.012, 0.000], device=dev)
            dir6 = tgt6 - rT
            dist6 = float(dir6.norm())
            step_w6 = quat_apply(s["cube_q"].unsqueeze(0), (dir6 / max(dist6, 1e-6) * 0.001).unsqueeze(0))[0]
            wp6 = s["wrist_p"].clone()
            print(f"     [PC6-2] 拇指合拢B 就位: thumb4=({rT[0]*100:+.1f},{rT[1]*100:+.1f},{rT[2]*100:+.1f})cm 盒距={dT6.norm().item()*1e3:.0f}mm"
                  f" → 滑入 {dist6*1000:.0f}mm（1mm/循环；Fth>0.10 或 cubeΔ>1mm 停）")
            for kk in range(min(int(dist6 / 0.001) + 1, 45)):
                wp6 = wp6 + step_w6
                for _ in range(4):
                    hold_wrist_step(wp6, wq0, hand6)
                s = snap()
                fth6 = s["forces"][0].item()
                dcb6 = (s["cube_p"] - cube_ref6).norm().item()
                if kk % 4 == 3 or fth6 > 0.05:
                    r6 = to_cube0(s["tips"]["thumb4"])
                    d6 = r6 - torch.clamp(r6, -half, half)
                    print(f"        滑{(kk+1)*1.0:4.1f}mm: thumb4=({r6[0]*100:+.1f},{r6[1]*100:+.1f},{r6[2]*100:+.1f})cm 盒距={d6.norm().item()*1e3:3.0f}mm"
                          f" Fth={fth6:.3f} Fmid={s['forces'][2]:.3f} cubeΔ={dcb6*1000:.1f}mm")
                if fth6 > 0.10:
                    print(f"        ✓ 拇指触到（Fth={fth6:.3f}N）")
                    break
                if dcb6 > 0.001:
                    print("        ⚠️ cube 被推 1mm → 停滑")
                    break
            # 3) 收口：手指加深 + 臂 +y 细进（看双侧力齐升 & cubeΔ 守住）
            print("     [PC6-3] 收口（手指 j2 加深 ≤0.3 + 臂 +y 0.3mm×10；Fmax>0.60N 或 cubeΔ>1.5mm 停）…")
            cube_ref6b = snap()["cube_p"].clone()
            wp6b = s["wrist_p"].clone()
            j2_cur = float(hand6[5])
            j2_end = min(1.6, j2_cur + 0.3)
            for kj in range(1, 9):
                hand6[5] = j2_cur + (j2_end - j2_cur) * kj / 8
                for _ in range(3):
                    hold_wrist_step(wp6b, wq0, hand6)
            s = snap()
            print(f"        手指加深后: Fmid={s['forces'][2]:.3f} Fth={s['forces'][0]:.3f} Fmax={float(s['forces'].max()):.3f}N cubeΔ={(s['cube_p']-cube_ref6b).norm().item()*1000:.1f}mm")
            y_dir6 = quat_apply(s["cube_q"].unsqueeze(0), torch.tensor([[0.0, 1.0, 0.0]], device=dev))[0]
            for kp6 in range(10):
                wp6b = wp6b + y_dir6 * 0.0003
                for _ in range(3):
                    hold_wrist_step(wp6b, wq0, hand6)
                s = snap()
                fmid6 = s["forces"][2].item()
                fth6 = s["forces"][0].item()
                fmax6 = float(s["forces"].max())
                dcb6 = (s["cube_p"] - cube_ref6b).norm().item()
                print(f"        +y{(kp6+1)*0.3:4.1f}mm: Fmid={fmid6:.3f} Fth={fth6:.3f} Fmax={fmax6:.3f}N cubeΔ={dcb6*1000:.1f}mm")
                if fmax6 > 0.60:
                    print("        ★ 力到 0.60N → 停收口")
                    break
                if dcb6 > 0.0015:
                    print("        ⚠️ cube 被推 1.5mm → 停收口")
                    break
            for _ in range(30):
                hold_wrist_step(wp6b, wq0, hand6)
            s = snap()
            rT = to_cube0(s["tips"]["thumb4"])
            rM = to_cube0(s["tips"]["middle4"])
            fth6 = s["forces"][0].item()
            fmid6 = s["forces"][2].item()
            dcb6 = (s["cube_p"] - cube_ref6).norm().item()
            print(f"     [PC6] 合围末态: thumb4=({rT[0]*100:+.1f},{rT[1]*100:+.1f},{rT[2]*100:+.1f})cm | mid4=({rM[0]*100:+.1f},{rM[1]*100:+.1f},{rM[2]*100:+.1f})cm"
                  f" Fth={fth6:.3f} Fmid={fmid6:.3f}N cubeΔ={dcb6*1000:.1f}mm")
            print(f"          判定: {'双侧有力且 cube 未动 → 楔住成立！' if (fth6 > 0.05 and fmid6 > 0.05 and dcb6 < 0.0015) else '未完全楔住（看上面数值）'}")
            # 4) 微提 +1mm 探测
            print("     [PC6-4] 微提 +1mm 探测（看 cube_z 跟不跟）…")
            z06 = s["cube_p"][2].item()
            wp6c = wp6b + torch.tensor([0.0, 0.0, 0.001], device=dev)
            for _ in range(12):
                hold_wrist_step(wp6c, wq0, hand6)
            s = snap()
            dz6 = s["cube_p"][2].item() - z06
            print(f"     [PC6-4] 结果: cube_z {dz6*1000:+.2f}mm Fth={s['forces'][0]:.3f} Fmid={s['forces'][2]:.3f}N"
                  f" → {'cube 跟动了！' if dz6 > 0.0002 else '没跟动（滑脱）'}")
            # 5) 松手回收
            hand6.zero_()
            for _ in range(20):
                step_hold_wrist(hand6)

        # ═════════════════ [PC7] v0.23：触到即冻 · 对夹轴重测（降手→深卷·冻→臂+y按压→拇指滑入·冻→-x/+y/关节 探针→微提）═════════════════
        #   v0.22 实测：①降手 3cm 生效——手指首获真接触（卷到 1.25 时 Fmid=0.519N）；但"停指后力全消"：
        #   冻前目标仍挂着 1.25，手指继续收敛→从棱上滚过去（到位 mid4 z=+0.5 → 末态 +3.6cm＝整段在后撤）。
        #   ②拇指触到 0.310N 后，沿滑入向续进力**单调衰减**（0.237→0.004，切向蹭走）；臂 +y 时拇指断续
        #   有力（0.04~0.19N）而手指 0。③结论：接触都能"碰到"，下一步 = **留住 + 找对夹轴**。
        #   本段核心：**触到即冻**（该关节目标改成"当前实际角"，杜绝收敛把接触滚丢）+ 逐轴探针表。
        if args_cli.pc7:
            jid7 = {nm: i for i, nm in enumerate(joint_names)}
            restore_cube()
            for _ in range(args_cli.cal_rest):
                step_hold_wrist()
            hand7 = torch.zeros(8, device=dev)
            cube_ref7 = snap()["cube_p"].clone()
            # a) 预降（全部伸直、无接触；守卫 Fmax>0.03 / cubeΔ>1mm）
            print(f"\n     [PC7-a] 预降手 {args_cli.pc7_drop*1000:.0f}mm（手指伸直；守卫 Fmax>0.03/cubeΔ>1mm 停）")
            wp7 = snap()["wrist_p"].clone()
            n_drop7 = int(args_cli.pc7_drop / 0.001)
            for kd in range(n_drop7):
                wp7 = wp7 + torch.tensor([0.0, 0.0, -0.001], device=dev)
                for _ in range(3):
                    hold_wrist_step(wp7, wq0, hand7)
                s = snap()
                fmax7 = float(s["forces"].max())
                dcb7 = (s["cube_p"] - cube_ref7).norm().item()
                if kd % 5 == 4:
                    print(f"        降{(kd+1):2d}mm: Fmax={fmax7:.3f}N cubeΔ={dcb7*1000:.1f}mm")
                if fmax7 > 0.03:
                    print(f"        ⚠️ 下降中触到东西（Fmax={fmax7:.3f}N）→ 停降")
                    break
                if dcb7 > 0.001:
                    print("        ⚠️ cube 被扰 → 停降")
                    break
            s = snap()
            rM7 = to_cube0(s["tips"]["middle4"])
            print(f"     [PC7-a] 降毕: 腕z={s['wrist_p'][2].item():.3f} mid4=({rM7[0]*100:+.1f},{rM7[1]*100:+.1f},{rM7[2]*100:+.1f})cm")
            # b) 手指深卷：触到即冻（目标改成触到时的实际角，杜绝继续收敛→滚走）
            print("     [PC7-b] 手指深卷（j2/j3 → 1.4；触到即冻；过卷保护；cubeΔ>2mm 停）")
            prev_box7 = 1e9
            rise7 = 0
            for kk in range(1, 29):
                tc = 1.4 * kk / 28
                hand7[5], hand7[6], hand7[7] = tc, tc, 0.30
                for _ in range(2):
                    hold_wrist_step(wp7, wq0, hand7)
                s = snap()
                r7 = to_cube0(s["tips"]["middle4"])
                d7 = r7 - torch.clamp(r7, -half, half)
                box7 = float(d7.norm())
                fm7 = s["forces"][2].item()
                dcb7 = (s["cube_p"] - cube_ref7).norm().item()
                if kk % 4 == 0 or fm7 > 0.02:
                    print(f"        j2/j3→{tc:.2f}: mid4=({r7[0]*100:+.1f},{r7[1]*100:+.1f},{r7[2]*100:+.1f})cm 盒距={box7*1e3:3.0f}mm Fmid={fm7:.3f}N cubeΔ={dcb7*1000:.1f}mm")
                if fm7 > 0.05:
                    a2 = s["jpos"][jid7["middle2_joint"]].item()
                    a3 = s["jpos"][jid7["middle3_joint"]].item()
                    a4 = s["jpos"][jid7["middle4_joint"]].item()
                    hand7[5], hand7[6], hand7[7] = a2, a3, a4
                    print(f"        ✓ 触到即冻（Fmid={fm7:.3f}N @ 目标{tc:.2f}；实际角 j2={a2:.2f} j3={a3:.2f} j4={a4:.2f} → 手指目标改冻在实际角）")
                    break
                if box7 < 0.050 and box7 > prev_box7 + 0.0005:
                    rise7 += 1
                    if rise7 >= 2:
                        print(f"        ✓ 已达最接近（盒距回升 @ {tc:.2f}；未触到力）")
                        break
                else:
                    rise7 = 0
                prev_box7 = box7
                if dcb7 > 0.002:
                    print("        ⚠️ cube 被推 → 停卷")
                    break
            # b2) 手指保力表：冻住后 20 步（每 5 步打印）——"真保持"还是"松手即散"
            for kb in range(1, 5):
                for _ in range(5):
                    hold_wrist_step(wp7, wq0, hand7)
                s = snap()
                r7 = to_cube0(s["tips"]["middle4"])
                d7 = r7 - torch.clamp(r7, -half, half)
                print(f"        [保力]{kb*5:3d}步: Fmid={s['forces'][2].item():.3f}N 盒距={d7.norm().item()*1e3:3.0f}mm mid4=({r7[0]*100:+.1f},{r7[1]*100:+.1f},{r7[2]*100:+.1f})cm")
            # c) 手指例行 finger_reach7：柔性退 0.90 → 腕回锁 → +0.05 逼近至触到即冻 → -0.08 回撤 → +0.02 从下方扫力
            def relock7(tag7):
                """v0.28：腕位姿回锁——把 wp7/wq0 的残差（漂移）用位姿伺服拉回去。
                v0.26 教训：e2 通道腕漂了 ~6cm/49°；v0.27 实测 12~28° 卡住不收——已配合
                hold_wrist_step 的方向保真修正。本版：预算 150→250、门限 rotErr<1°→<2°、每 50 步打进度。"""
                s0r = snap()
                for it in range(250):
                    s = snap()
                    e_p = float((wp7 - s["wrist_p"]).norm())
                    e_r = float(torch.rad2deg(_rotvec(_qmul(wq0, _qconj(s["wrist_q"]))).norm()))
                    if it % 50 == 49:
                        print(f"        [{tag7}] 回锁中… |e_p|={e_p*1000:.1f}mm rotErr={e_r:.1f}° @ {it+1} 步")
                    if e_p < 0.002 and e_r < 2.0:
                        print(f"        [{tag7}] 腕回锁 ✓ |e_p|={e_p*1000:.1f}mm rotErr={e_r:.1f}° @ {it} 步"
                              f"（Fth {s0r['forces'][0].item():.3f}→{s['forces'][0].item():.3f} Fmid {s0r['forces'][2].item():.3f}→{s['forces'][2].item():.3f}）")
                        return
                    hold_wrist_step(wp7, wq0, hand7)
                s = snap()
                e_p = float((wp7 - s["wrist_p"]).norm())
                e_r = float(torch.rad2deg(_rotvec(_qmul(wq0, _qconj(s["wrist_q"]))).norm()))
                print(f"        [{tag7}] ⚠️ 腕回锁未收敛 |e_p|={e_p*1000:.1f}mm rotErr={e_r:.1f}°（继续）")

            def finger_reach7(tag7):
                """v0.27：手指"重新到达+从下方找力"（c 拇指前 / e2 拇指后复用）。
                柔性退 0.90 → **腕回锁** → +0.05 逼近至触到即冻（上限 1.45）→ -0.08 回撤 →
                +0.02 级扫描（首达 Fmid>0.10 停保留；峰后跌零回峰位）。cube 守卫 1mm。"""
                for tgt7v in (1.05, 1.00, 0.95, 0.90):
                    hand7[5], hand7[6], hand7[7] = tgt7v, tgt7v, 0.30
                    for _ in range(4):
                        hold_wrist_step(wp7, wq0, hand7)
                for _ in range(10):
                    hold_wrist_step(wp7, wq0, hand7)
                relock7(tag7)
                touched7 = False
                for kk in range(1, 14):
                    hand7[5] = min(float(hand7[5]) + 0.05, 1.45)
                    hand7[6] = min(float(hand7[6]) + 0.05, 1.45)
                    for _ in range(2):
                        hold_wrist_step(wp7, wq0, hand7)
                    s = snap()
                    fm = s["forces"][2].item()
                    if fm > 0.05:
                        a2 = s["jpos"][jid7["middle2_joint"]].item()
                        a3 = s["jpos"][jid7["middle3_joint"]].item()
                        hand7[5], hand7[6] = a2, a3
                        print(f"        [{tag7}] 触到即冻：Fmid={fm:.3f}（实际角 j2={a2:.2f} j3={a3:.2f} → 目标改冻）")
                        touched7 = True
                        break
                    if (s["cube_p"] - cube_ref7).norm().item() > 0.001:
                        print(f"        [{tag7}] ⚠️ cube 被推 → 停")
                        return
                if not touched7:
                    s = snap()
                    e_p = float((wp7 - s["wrist_p"]).norm())
                    e_r = float(torch.rad2deg(_rotvec(_qmul(wq0, _qconj(s["wrist_q"]))).norm()))
                    r7 = to_cube0(s["tips"]["middle4"])
                    d7 = r7 - torch.clamp(r7, -half, half)
                    print(f"        [{tag7}] ⚠️ 逼近未触到 → 直接报告（不细扫）: 盒距={d7.norm().item()*1e3:.0f}mm j2实={s['jpos'][jid7['middle2_joint']].item():.2f}"
                          f" |e_p|={e_p*1000:.1f}mm rotErr={e_r:.1f}° Fth={s['forces'][0].item():.3f} Fmid={s['forces'][2].item():.3f}")
                    return
                hand7[5] = float(hand7[5]) - 0.08
                hand7[6] = float(hand7[6]) - 0.08
                for _ in range(8):
                    hold_wrist_step(wp7, wq0, hand7)
                peak_f, peak_k = 0.0, 0
                for kk in range(1, 15):
                    hand7[5] = float(hand7[5]) + 0.02
                    hand7[6] = float(hand7[6]) + 0.02
                    for _ in range(3):
                        hold_wrist_step(wp7, wq0, hand7)
                    s = snap()
                    fm = s["forces"][2].item()
                    dcb = (s["cube_p"] - cube_ref7).norm().item()
                    r7 = to_cube0(s["tips"]["middle4"])
                    d7 = r7 - torch.clamp(r7, -half, half)
                    print(f"        [{tag7}]{kk:2d}: Fmid={fm:.3f} Fth={s['forces'][0].item():.3f} 盒距={d7.norm().item()*1e3:3.0f}mm j2实={s['jpos'][jid7['middle2_joint']].item():.2f} cubeΔ={dcb*1000:.1f}mm")
                    if fm > peak_f:
                        peak_f, peak_k = fm, kk
                    if fm > 0.10:
                        print(f"        [{tag7}] ★ Fmid={fm:.3f} @ +{kk*0.02:.2f}（从下方着力）→ 停保留")
                        break
                    if peak_f > 0.04 and fm < 0.03 and (kk - peak_k) >= 2:
                        bk = (kk - peak_k) * 0.02
                        hand7[5] = float(hand7[5]) - bk
                        hand7[6] = float(hand7[6]) - bk
                        for _ in range(6):
                            hold_wrist_step(wp7, wq0, hand7)
                        print(f"        [{tag7}] ⚠️ 峰(Fmid={peak_f:.3f}@+{peak_k*0.02:.2f})后跌零 → 回峰位停")
                        break
                    if dcb > 0.001:
                        print(f"        [{tag7}] ⚠️ cube 被推 1mm → 停")
                        break
                for _ in range(10):
                    hold_wrist_step(wp7, wq0, hand7)
                s = snap()
                print(f"        [{tag7}] 保持后: Fmid={s['forces'][2].item():.3f} Fth={s['forces'][0].item():.3f} cubeΔ={(s['cube_p']-cube_ref7).norm().item()*1000:.1f}mm")

            print("     [PC7-c] 手指例行 finger_reach7(c)：退0.90→逼近触到→回撤→+0.02 从下方扫着力")
            finger_reach7("c")
            relock7("d-pre")   # v0.28: 拇指相位前再清一次腕残差（c 的快速动作会攒漂移）
            # d) 拇指合拢B + 滑入 → 触到即冻 → 保力表
            print("     [PC7-d] 拇指合拢B（30 步）→ 滑入（0.5mm/循环；Fth>0.08 触到即冻）")
            s = snap()
            fmid_pre7 = s["forces"][2].item()
            hand7[:4] = torch.tensor([1.0, 1.0, 0.8, 0.8], device=dev)
            for _ in range(30):
                hold_wrist_step(wp7, wq0, hand7)
            s = snap()
            rT = to_cube0(s["tips"]["thumb4"])
            dT7 = rT - torch.clamp(rT, -half, half)
            tgt7 = torch.tensor([0.000, half + 0.012, 0.000], device=dev)
            dir7 = tgt7 - rT
            dist7 = float(dir7.norm())
            step_w7 = quat_apply(s["cube_q"].unsqueeze(0), (dir7 / max(dist7, 1e-6) * 0.001).unsqueeze(0))[0]
            print(f"        合拢B 就位: thumb4=({rT[0]*100:+.1f},{rT[1]*100:+.1f},{rT[2]*100:+.1f})cm 盒距={dT7.norm().item()*1e3:.0f}mm"
                  f" | 交叉影响: Fmid {fmid_pre7:.3f}→{s['forces'][2].item():.3f}N | 滑入 {dist7*1000:.0f}mm（0.5mm/循环慢滑）")
            cube_ref7d = snap()["cube_p"].clone()   # v0.25: 滑入段独立基线（防陈旧位移误触守卫）
            for kk in range(150):
                wp7 = wp7 + step_w7 * 0.5
                for _ in range(3):
                    hold_wrist_step(wp7, wq0, hand7)
                s = snap()
                fth7 = s["forces"][0].item()
                dcb7 = (s["cube_p"] - cube_ref7d).norm().item()
                r7 = to_cube0(s["tips"]["thumb4"])
                d7 = r7 - torch.clamp(r7, -half, half)
                if kk % 8 == 7 or fth7 > 0.02:
                    print(f"        滑{(kk+1)*0.5:5.1f}mm: thumb4=({r7[0]*100:+.1f},{r7[1]*100:+.1f},{r7[2]*100:+.1f})cm 盒距={d7.norm().item()*1e3:3.0f}mm Fth={fth7:.3f} Fmid={s['forces'][2].item():.3f} cubeΔ={dcb7*1000:.1f}mm")
                if fth7 > 0.08:
                    a1 = s["jpos"][jid7["thumb1_joint"]].item()
                    a2 = s["jpos"][jid7["thumb2_joint"]].item()
                    a3 = s["jpos"][jid7["thumb3_joint"]].item()
                    a4 = s["jpos"][jid7["thumb4_joint"]].item()
                    hand7[0], hand7[1], hand7[2], hand7[3] = -a1, -a2, a3, a4
                    print(f"        ✓ 拇指触到即冻（Fth={fth7:.3f}N @ 滑{(kk+1)*0.5:.1f}mm；实际角 j=({a1:+.2f},{a2:+.2f},{a3:+.2f},{a4:+.2f}) → 目标改冻）")
                    break
                if dcb7 > 0.0008:
                    print(f"        ⚠️ cube 被推 {dcb7*1000:.1f}mm（滑{(kk+1)*0.5:.1f}mm、盒距={d7.norm().item()*1e3:.0f}mm）→ 停滑")
                    break
            # d2) 拇指保力表：冻住后 20 步
            for kb in range(1, 5):
                for _ in range(5):
                    hold_wrist_step(wp7, wq0, hand7)
                s = snap()
                r7 = to_cube0(s["tips"]["thumb4"])
                d7 = r7 - torch.clamp(r7, -half, half)
                print(f"        [保力]{kb*5:3d}步: Fth={s['forces'][0].item():.3f}N 盒距={d7.norm().item()*1e3:3.0f}mm thumb4=({r7[0]*100:+.1f},{r7[1]*100:+.1f},{r7[2]*100:+.1f})cm Fmid={s['forces'][2].item():.3f} cubeΔ={(s['cube_p']-cube_ref7d).norm().item()*1000:.1f}mm")
            # e) 拇指收口（正式）：关节微捏 t3/t4 +0.04（力停 0.15N；v0.27: e 前先腕回锁）
            relock7("e-pre")
            cube_ref7e = snap()["cube_p"].clone()
            print("     [PC7-e] 拇指收口：t3/t4 目标 +0.04×8（Fth>0.15 或 cubeΔ(段)>1.5mm 停）")
            for km in range(1, 9):
                hand7[2] = hand7[2] + 0.04
                hand7[3] = hand7[3] + 0.04
                for _ in range(3):
                    hold_wrist_step(wp7, wq0, hand7)
                s = snap()
                fth7 = s["forces"][0].item()
                dcb7 = (s["cube_p"] - cube_ref7e).norm().item()
                r7 = to_cube0(s["tips"]["thumb4"])
                d7 = r7 - torch.clamp(r7, -half, half)
                print(f"        捏{km*0.04:4.2f}: Fth={fth7:.3f} Fmid={s['forces'][2].item():.3f} 盒距={d7.norm().item()*1e3:3.0f}mm cubeΔ(段)={dcb7*1000:.1f}mm")
                if fth7 > 0.15:
                    print("        ★ 拇指力到 0.15N → 停（目标保留，后续自然收敛会继续爬升）")
                    break
                if dcb7 > 0.0015:
                    print("        ⚠️ cube 被推 >1.5mm（本段）→ 停")
                    break
            for _ in range(15):
                hold_wrist_step(wp7, wq0, hand7)
            s = snap()
            print(f"     [PC7-e] 收口保持 15 步: Fth={s['forces'][0].item():.3f} Fmid={s['forces'][2].item():.3f} cubeΔ(段)={(s['cube_p']-cube_ref7e).norm().item()*1000:.1f}mm")
            # e2) 拇指收口后，手指重新"到达+找力"（复用例行；验证拇指相位后能否再建立手指接触）
            print("     [PC7-e2] 拇指收口后手指重做 finger_reach7(e2)（Fth 列看交叉影响）")
            finger_reach7("e2")
            s = snap()
            e_p7 = float((wp7 - s["wrist_p"]).norm())
            e_r7 = float(torch.rad2deg(_rotvec(_qmul(wq0, _qconj(s["wrist_q"]))).norm()))
            print(f"     [PC7] 腕位姿诊断: |e_p|={e_p7*1000:.1f}mm rotErr={e_r7:.1f}°（>5mm/3° = 本轮结果存疑）")
            rT = to_cube0(s["tips"]["thumb4"])
            rM = to_cube0(s["tips"]["middle4"])
            fth7 = s["forces"][0].item()
            fmid7 = s["forces"][2].item()
            dcb7 = (s["cube_p"] - cube_ref7).norm().item()
            print(f"     [PC7] 合围末态: thumb4=({rT[0]*100:+.1f},{rT[1]*100:+.1f},{rT[2]*100:+.1f})cm | mid4=({rM[0]*100:+.1f},{rM[1]*100:+.1f},{rM[2]*100:+.1f})cm"
                  f" Fth={fth7:.3f} Fmid={fmid7:.3f}N cubeΔ={dcb7*1000:.1f}mm")
            print(f"          判定: {'双侧有力且 cube 未动 → 楔住成立！' if (fth7 > 0.05 and fmid7 > 0.05 and dcb7 < 0.0015) else '未完全楔住（看上面数值）'}")
            # f) 微提 +1mm 探测
            print("     [PC7-f] 微提 +1mm 探测…")
            z07 = s["cube_p"][2].item()
            wp7c = wp7 + torch.tensor([0.0, 0.0, 0.001], device=dev)
            for _ in range(12):
                hold_wrist_step(wp7c, wq0, hand7)
            s = snap()
            dz7 = s["cube_p"][2].item() - z07
            fth_v7 = s["forces"][0].item()
            fmid_v7 = s["forces"][2].item()
            if dz7 > 0.0002 and (fth_v7 > 0.03 or fmid_v7 > 0.03):
                v7 = "cube 跟动 + 有接触力 → 楔住!"
            elif dz7 > 0.0002:
                v7 = "cube 动了但力读数≈0 → 疑似蹭动/回弹，不算稳楔"
            else:
                v7 = "没跟动（滑脱）"
            print(f"     [PC7-f] 结果: cube_z {dz7*1000:+.2f}mm Fth={fth_v7:.3f} Fmid={fmid_v7:.3f}N → {v7}")
            # g) 松手回收
            hand7.zero_()
            for _ in range(20):
                step_hold_wrist(hand7)

        # ═════════════════ PC3：夹取实战 v0.12（[1]预置 → [2]力停合拢 → [4]中指扫入捏持 → [5A]补压 0.9 → [6]保压慢提）═════════════════
        if args_cli.rehearsal:
            fmid_gate = args_cli.press_force      # 中指触面阈值（传感器若醒）
            fth_touch = args_cli.thumb_touch      # 拇指贴面力停（N）
            fth_grip = args_cli.squeeze_force     # 双侧压紧目标（拇指侧 N）

            def box_dist(p):
                """指尖参考点→cube 表面盒距（m）。≈12mm 时真实表面贴合（v0.6 实测标定）。"""
                q = to_cube0(p)
                d = q - torch.clamp(q, -half, half)
                return float(d.norm())

            restore_cube()                        # v0.18: PC3 也从 cube 出生位开始（防 PC2 推挤残留）
            for _ in range(15):                   # v0.19: 复位后静置（防"cube 落稳位移"误触 [2] 的 cubeΔ 守卫）
                hold_wrist_step(wp0, wq0)
            cube_start = snap()["cube_p"].clone()
            cube_ref = cube_start.clone()
            hand = torch.zeros(8, device=dev)
            print(f"\n     ══ PC3 v0.28：stage_x={args_cli.stage_x*1000:+.0f}mm，手部μ={args_cli.hand_friction:.2f}，手部clip={args_cli.hand_clip:.1f}，engage={'是' if engage_done else '否'}，口心对位={'开' if args_cli.mouth_align else '关'}，加压={'开' if args_cli.press else '关'} ══")
            wp_t = wp0.clone()
            # [1] 臂 -x 预置（手全开、无接触）
            st = 0.0015
            n_st = int(abs(args_cli.stage_x) / st + 1e-9)
            dirx = -1.0 if args_cli.stage_x < 0 else 1.0
            for _ in range(n_st):
                wp_t = wp_t + torch.tensor([dirx * st, 0.0, 0.0], device=dev)
                for _ in range(4):
                    hold_wrist_step(wp_t, wq0)
            s = snap()
            r = to_cube0(s["tips"]["thumb4"])
            print(f"     [1] 预置完成: thumb4=({r[0]:+.3f},{r[1]:+.3f},{r[2]:+.3f}) 盒距 mid4={box_dist(s['tips']['middle4'])*1000:.0f}mm"
                  f" thumb4={box_dist(s['tips']['thumb4'])*1000:.0f}mm cubeΔ={(s['cube_p']-cube_ref).norm().item()*1000:.1f}mm")
            # [2] 拇指合拢（v0.8：力停——每 2 步采样，触到 0.8×thumb_touch 即停；cubeΔ 兜底）
            th_hit = False
            th_stop_t = 1.0
            fth_c = fth_touch * 0.8
            for k in range(1, 61):
                t = k / 60
                hand[0], hand[1], hand[2], hand[3] = 1.0 * t, -1.0 * t, 0.8 * t, 0.8 * t
                hold_wrist_step(wp_t, wq0, hand)
                if k % 2 == 0:
                    s = snap()
                    fth = s["forces"][0].item()
                    dcb = (s["cube_p"] - cube_ref).norm().item()
                    if fth > fth_c:
                        th_hit = True
                        th_stop_t = t
                        print(f"     [2] 合拢触面 @ t={t:.2f}（Fthumb={fth:.3f}N）→ 停合拢（保留当前合拢量）")
                        break
                    if dcb > 0.0006:
                        th_stop_t = t
                        print(f"     [2] ⚠️ 合拢中 cube 被推 {dcb*1000:.1f}mm（Fthumb={fth:.3f}N）→ 停合拢")
                        break
            s = snap()
            r = to_cube0(s["tips"]["thumb4"])
            print(f"     [2] 合拢停至 t={th_stop_t:.2f}: thumb4=({r[0]:+.3f},{r[1]:+.3f},{r[2]:+.3f})"
                  f" 盒距={box_dist(s['tips']['thumb4'])*1000:.0f}mm Fthumb={s['forces'][0]:.4f}N cubeΔ={(s['cube_p']-cube_ref).norm().item()*1000:.1f}mm")
            cube_ref = s["cube_p"].clone()          # v0.8: 重新基线（防"陈旧位移"误触后续守卫）
            # [3] 臂 -y 慢推 ≤18mm（仅当 [2] 未触；v0.14：engage 已对准时自动跳过）
            if (not th_hit) and (not engage_done):
                for k in range(12):
                    wp_t = wp_t + torch.tensor([0.0, -0.0015, 0.0], device=dev)
                    for _ in range(4):
                        hold_wrist_step(wp_t, wq0, hand)
                    s = snap()
                    fth = s["forces"][0].item()
                    r = to_cube0(s["tips"]["thumb4"])
                    dth = box_dist(s["tips"]["thumb4"])
                    dcb = (s["cube_p"] - cube_ref).norm().item()
                    print(f"     [3] -y{(k+1)*1.5:4.1f}mm: Fthumb={fth:.4f}N thumb4=({r[0]:+.3f},{r[1]:+.3f},{r[2]:+.3f}) 盒距={dth*1000:3.0f}mm cubeΔ={dcb*1000:.1f}mm")
                    if fth > fth_touch:
                        th_hit = True
                        print(f"     [3] → 拇指贴面 ✓（Fthumb={fth:.3f}N，盒距={dth*1000:.0f}mm）")
                        break
                    if dcb > 0.001:
                        print(f"     [3] ⚠️ cube 被推 {dcb*1000:.1f}mm → 停")
                        break
                if not th_hit:
                    print("     [3] → 拇指未触满 18mm——继续，看中指把它顶入拇指时 Fthumb 是否上升")
                cube_ref = snap()["cube_p"].clone()  # v0.8: 再基线
            # [4] 中指扫入：卷曲 j2/j3 → 1.0（参考点再向面心扫 ~15-20mm），双侧压紧判 Fthumb>fth_grip
            squeeze_ok = False
            mid_touched = False
            side_note = False
            for k in range(1, 61):
                t = k / 60
                hand[5], hand[6], hand[7] = 1.0 * t, 1.0 * t, 0.15
                hold_wrist_step(wp_t, wq0, hand)
                if k % 2 == 0:
                    s = snap()
                    fmid = s["forces"][2].item()
                    fth = s["forces"][0].item()
                    dcb = (s["cube_p"] - cube_ref).norm().item()
                    if fth > fth_grip and fmid > 0.03:
                        squeeze_ok = True
                        print(f"     [4] ★ 双侧压紧成立 @ t={t:.2f}：Fthumb={fth:.3f}N（Fmid={fmid:.3f}）cubeΔ={dcb*1000:.1f}mm")
                        break
                    if fth > fth_grip and fmid <= 0.03 and not side_note:
                        side_note = True
                        print(f"     [4] 注意：拇指侧 Fthumb={fth:.3f}N 已过阈值但 Fmid≈0 —— 单侧接触，不算压紧（v0.16 判定）")
                    if fmid > fmid_gate and not mid_touched:
                        mid_touched = True
                        print(f"     [4] 中指触面（传感器醒来）@ t={t:.2f}：Fmid={fmid:.3f}N")
                    if fmid > 0.5:
                        print(f"     [4] ⚠️ 中指力过大 {fmid:.2f}N → 停")
                        break
                    if dcb > 0.0035:
                        print(f"     [4] ⚠️ cube 被推 {dcb*1000:.1f}mm → 停")
                        break
                if k % 10 == 0:
                    s = snap()
                    rm = to_cube0(s["tips"]["middle4"])
                    print(f"     [4] t={t:.2f}: mid4=({rm[0]:+.3f},{rm[1]:+.3f},{rm[2]:+.3f}) 盒距={box_dist(s['tips']['middle4'])*1000:3.0f}mm"
                          f" Fmid={s['forces'][2]:.3f} Fthumb={s['forces'][0]:.3f}N cubeΔ={(s['cube_p']-cube_ref).norm().item()*1000:.1f}mm")
            # [B2] v0.16：口心对位接近——把"闭合口"（拇指尖-中指尖 中点）套到 cube 质心（申报式平移、全守卫）
            if args_cli.mouth_align:
                s = snap()
                T_tip = to_cube0(s["tips"]["thumb4"]).clone()
                M_tip = to_cube0(s["tips"]["middle4"]).clone()
                mid_t = 0.5 * (T_tip + M_tip)
                L_mid = float(mid_t.norm())
                if L_mid < 0.004:
                    print("     [B2] 口心已在质心附近（<4mm），跳过对位")
                else:
                    dir_u = -mid_t / L_mid
                    slide_v = dir_u * min(L_mid + 0.010, 0.100)      # 含 1cm 预压量，上限 10cm
                    slide_w = quat_apply(s["cube_q"].unsqueeze(0), slide_v.unsqueeze(0))[0]
                    w_goal = s["wrist_p"] + slide_w
                    print(f"     [B2] 口心对位接近：口心=({mid_t[0]*100:+.1f},{mid_t[1]*100:+.1f},{mid_t[2]*100:+.1f})cm"
                          f" → 平移量 ({slide_v[0]*100:+.1f},{slide_v[1]*100:+.1f},{slide_v[2]*100:+.1f})cm（慢速；Fmax>0.10N / cubeΔ>1.5mm 停）")
                    th_old = th_stop_t
                    th_stop_t = max(0.35, th_stop_t * 0.66)
                    hand[0], hand[1], hand[2], hand[3] = 1.0 * th_stop_t, -1.0 * th_stop_t, 0.8 * th_stop_t, 0.8 * th_stop_t
                    for _ in range(12):
                        hold_wrist_step(wp_t, wq0, hand)
                    print(f"     [B2] 拇指先松到 t={th_stop_t:.2f}（原 {th_old:.2f}）防拖拽")
                    cube_refB = snap()["cube_p"].clone()
                    for kb in range(50):
                        s = snap()
                        e_b = w_goal - s["wrist_p"]
                        fmax_b = float(s["forces"].max())
                        dcb = (s["cube_p"] - cube_refB).norm().item()
                        if kb % 5 == 0:
                            tT = to_cube0(s["tips"]["thumb4"]); tM = to_cube0(s["tips"]["middle4"])
                            tm_ = 0.5 * (tT + tM)
                            print(f"     [B2] it{kb:2d} 口心=({tm_[0]*100:+.1f},{tm_[1]*100:+.1f},{tm_[2]*100:+.1f})"
                                  f" |e|={e_b.norm().item()*1000:4.1f}mm Fmax={fmax_b:.3f}N cubeΔ={dcb*1000:.1f}mm")
                        if fmax_b > 0.10:
                            print(f"     [B2] ✓ 触到（Fmax={fmax_b:.3f}N）@ {kb} 步 → 停平移")
                            break
                        if dcb > 0.0015:
                            print(f"     [B2] ⚠️ cube 被推 {dcb*1000:.1f}mm → 停平移")
                            break
                        if e_b.norm().item() < 0.004:
                            print(f"     [B2] ✓ 平移到位（|e|={e_b.norm().item()*1000:.1f}mm）@ {kb} 步")
                            break
                        wp_t = s["wrist_p"] + torch.clamp(e_b, -0.004, 0.004)
                        for _ in range(5):
                            hold_wrist_step(wp_t, wq0, hand)
            # [5] 加力到位（v0.10：[4] 末 Fthumb 若不到 --squeeze_top → "拇指续合"补压；臂 +y 只作 B 计划）
            s = snap()
            fmid4 = s["forces"][2].item()
            fth4 = s["forces"][0].item()
            print(f"     [5] [4]末状态: Fmid={fmid4:.3f} Fthumb={fth4:.3f}N（补压目标 {args_cli.squeeze_top:.2f}N）")
            # Stage A：拇指续合补压（v0.17：--press 门控）
            if args_cli.press and th_stop_t < 0.995 and fth4 < args_cli.squeeze_top:
                for k in range(26):
                    th_stop_t = min(1.0, th_stop_t + 0.018)
                    hand[0], hand[1], hand[2], hand[3] = 1.0 * th_stop_t, -1.0 * th_stop_t, 0.8 * th_stop_t, 0.8 * th_stop_t
                    for _ in range(3):
                        hold_wrist_step(wp_t, wq0, hand)
                    s = snap()
                    fth = s["forces"][0].item()
                    fmid = s["forces"][2].item()
                    dcb = (s["cube_p"] - cube_ref).norm().item()
                    print(f"     [5A] 补压 t→{th_stop_t:.2f}: Fmid={fmid:.3f} Fthumb={fth:.3f}N cubeΔ={dcb*1000:.1f}mm")
                    if fth > args_cli.squeeze_top:
                        squeeze_ok = True
                        print(f"     [5A] ★ 加力达标：Fthumb={fth:.3f}N（Fmid={fmid:.3f}）")
                        break
                    if fmid > 0.9:
                        print(f"     [5A] ⚠️ 中指力过大 {fmid:.2f}N → 停")
                        break
                    if dcb > 0.0015:
                        print("     [5A] ⚠️ cube 被推 1.5mm → 停")
                        break
            # Stage B：臂 +y 微爬 0.25mm 级（v0.17：--press 门控）
            if (not squeeze_ok) and args_cli.press:
                for k in range(6):
                    wp_t = wp_t + torch.tensor([0.0, 0.00025, 0.0], device=dev)
                    for _ in range(4):
                        hold_wrist_step(wp_t, wq0, hand)
                    s = snap()
                    fth = s["forces"][0].item()
                    fmid = s["forces"][2].item()
                    dcb = (s["cube_p"] - cube_ref).norm().item()
                    print(f"     [5B] +y{(k+1)*0.25:4.2f}mm: Fmid={fmid:.3f} Fthumb={fth:.3f}N cubeΔ={dcb*1000:.1f}mm")
                    if fth > args_cli.squeeze_top:
                        squeeze_ok = True
                        print(f"     [5B] ★ 加力达标：Fthumb={fth:.3f}N")
                        break
                    if fmid > 0.9:
                        print("     [5B] ⚠️ 中指力过大 → 停")
                        break
                    if dcb > 0.0015:
                        print("     [5B] ⚠️ cube 被推 1.5mm → 停")
                        break
            # [6] 接触态报告 +（v0.17：加压/微提段由 --press 门控；默认 0=只观察姿势）
            for _ in range(8):
                hold_wrist_step(wp_t, wq0, hand)
            s = snap()
            fmid = s["forces"][2].item()
            fth = s["forces"][0].item()
            dcb = (s["cube_p"] - cube_ref).norm().item()
            rm = to_cube0(s["tips"]["middle4"])
            r = to_cube0(s["tips"]["thumb4"])
            nmid = quat_apply_inverse(s["cube_q"].unsqueeze(0), quat_apply(robot.data.body_link_quat_w[0, body_id["middle4"]].unsqueeze(0), torch.tensor([[0.0, 0.0, -1.0]], device=dev)))[0]
            nth = quat_apply_inverse(s["cube_q"].unsqueeze(0), quat_apply(robot.data.body_link_quat_w[0, body_id["thumb4"]].unsqueeze(0), torch.tensor([[0.0, 0.0, -1.0]], device=dev)))[0]
            print(f"     [6] 接触态: Fmid={fmid:.3f} Fthumb={fth:.3f}N cubeΔ(本段)={dcb*1000:.1f}mm 总Δ={(s['cube_p']-cube_start).norm().item()*1000:.1f}mm"
                  f" mid4=({rm[0]:+.3f},{rm[1]:+.3f},{rm[2]:+.3f})(盒距{box_dist(s['tips']['middle4'])*1000:.0f}mm)"
                  f" thumb4=({r[0]:+.3f},{r[1]:+.3f},{r[2]:+.3f})")
            print(f"          指腹法线(cube系): mid4 pad=({nmid[0]:+.2f},{nmid[1]:+.2f},{nmid[2]:+.2f}) thumb4 pad=({nth[0]:+.2f},{nth[1]:+.2f},{nth[2]:+.2f})")
            if not args_cli.press:
                print("     [6] v0.17：--press 0（姿势专项）——不加压/不提起；保持合拢姿势 150 步供 GUI 观察…")
                for ko in range(150):
                    hold_wrist_step(wp_t, wq0, hand)
                    if ko % 50 == 49:
                        so = snap()
                        rTo = to_cube0(so["tips"]["thumb4"])
                        rMo = to_cube0(so["tips"]["middle4"])
                        nTo = quat_apply_inverse(so["cube_q"].unsqueeze(0), quat_apply(robot.data.body_link_quat_w[0, body_id["thumb4"]].unsqueeze(0), torch.tensor([[0.0, 0.0, -1.0]], device=dev)))[0]
                        print(f"          观察{ko+1:3d}: thumb4=({rTo[0]*100:+.1f},{rTo[1]*100:+.1f},{rTo[2]*100:+.1f})cm pad=({nTo[0]:+.2f},{nTo[1]:+.2f},{nTo[2]:+.2f})"
                              f" mid4=({rMo[0]*100:+.1f},{rMo[1]*100:+.1f},{rMo[2]*100:+.1f})cm"
                              f" Fmax={float(so['forces'].max()):.3f}N cubeΔ={(so['cube_p']-cube_ref).norm().item()*1000:.1f}mm")
            else:
                # [6b] 起飞前收尾补压（防静置衰减；至多 3 档 +0.01）
                for _ in range(3):
                    if fth < 0.55 and th_stop_t < 0.995:
                        th_stop_t = min(1.0, th_stop_t + 0.01)
                        hand[0], hand[1], hand[2], hand[3] = 1.0 * th_stop_t, -1.0 * th_stop_t, 0.8 * th_stop_t, 0.8 * th_stop_t
                        for _ in range(3):
                            hold_wrist_step(wp_t, wq0, hand)
                        s = snap()
                        fmid = s["forces"][2].item()
                        fth = s["forces"][0].item()
                        dcb = (s["cube_p"] - cube_ref).norm().item()
                        print(f"     [6b] 起飞补压 t→{th_stop_t:.2f}: Fmid={fmid:.3f} Fthumb={fth:.3f}N cubeΔ={dcb*1000:.1f}mm")
                        if dcb > 0.002:
                            print("     [6b] ⚠️ cube 被推 2mm → 停")
                            break
                    else:
                        break
                z0 = s["cube_p"][2].item()
                print("     [6] 微提 +2.5mm 慢提（0.5mm×5，每步打 cube_z/wrist_z）…")
                for j in range(5):
                    wp_t = wp_t + torch.tensor([0.0, 0.0, 0.0005], device=dev)
                    for _ in range(6):
                        hold_wrist_step(wp_t, wq0, hand)
                    s2 = snap()
                    print(f"        提{j+1}: cube_z={s2['cube_p'][2].item():.4f}({(s2['cube_p'][2].item()-z0)*1000:+.2f}mm)"
                          f" wrist_z={s2['wrist_p'][2].item():.4f} Fmid={s2['forces'][2]:.3f} Fthumb={s2['forces'][0]:.3f}N")
                    if j % 2 == 1 and s2["forces"][0].item() < 0.5 and th_stop_t < 0.995:
                        th_stop_t = min(1.0, th_stop_t + 0.008)
                        hand[0], hand[1], hand[2], hand[3] = 1.0 * th_stop_t, -1.0 * th_stop_t, 0.8 * th_stop_t, 0.8 * th_stop_t
                        for _ in range(3):
                            hold_wrist_step(wp_t, wq0, hand)
                s2 = snap()
                dz = s2["cube_p"][2].item() - z0
                fmid_v = s2["forces"][2].item()
                fth_v = s2["forces"][0].item()
                gripped = (fmid_v > 0.05) and (fth_v > 0.05)   # v0.16: 双侧都要有接触力才认夹持
                if dz > 0.001 and gripped:
                    verdict = "★ 提起成功（保持夹持）"
                elif dz > 0.0003 and gripped:
                    verdict = "部分提起"
                elif dz > 0.001:
                    verdict = "⚠️ cube_z 动了但零接触力——推动/回弹噪声，不算提起（v0.15 修正误报）"
                else:
                    verdict = "未提起"
                print(f"     [6] 微提判决: cube_z {dz*1000:+.2f}mm Fmid={fmid_v:.2f} Fthumb={fth_v:.2f}N  [{verdict}]")
                print("     [6] 保持 300 步观察（GUI 请看：中指面贴住哪、拇指顶在哪、cube 有没有跟着走）…")
                for kk in range(300):
                    hold_wrist_step(wp_t, wq0, hand)
                    if kk % 100 == 99:
                        s3 = snap()
                        print(f"          保持{kk+1:3d}: Fmid={s3['forces'][2]:.3f} Fthumb={s3['forces'][0]:.3f}N"
                              f" cubeΔ={(s3['cube_p']-cube_ref).norm().item()*1000:.1f}mm cube_z={s3['cube_p'][2].item():.4f}")
                s3 = snap()
                dz_hold = s3["cube_p"][2].item() - z0
                print(f"     [6] 保持段末判定: cube_z 相对起飞前 {dz_hold*1000:+.2f}mm → 真提起={'是' if dz_hold > 0.001 else '否'}（v0.16：保持期内 cube 未落回才算）")
            # [7] 松开 → 回原站位
            hand.zero_()
            for _ in range(20):
                hold_wrist_step(wp_t, wq0, hand)
            for _ in range(40):
                hold_wrist_step(wp0, wq0)
            print(f"     [7] 松开回收完成。握持成立={'是' if squeeze_ok else '否'}")

    # ═════════════════ PH：站姿保持（腕位姿保持，供 GUI 观察）═════════════════
    if args_cli.hold_end > 0:
        print(f"\n[PH] 保持 {args_cli.hold_end} 步（腕位姿保持）——GUI 请看：四指（相对 cube）贴/对着哪一面？拇指在 cube 的哪一侧、离多近？")
        for _ in range(args_cli.hold_end):
            hold_wrist_step(wp_stage, q_hold)

    # ═════════════════ PR：几何报告 ═════════════════
    sR = snap()
    cq, cp = sR["cube_q"], sR["cube_p"]

    def to_cube(p):
        return quat_apply_inverse(cq.unsqueeze(0), (p - cp).unsqueeze(0))[0]

    print(f"\n[PR] 几何报告（cube 局部系：面在 ±{half*1000:.0f} mm；-y 面外侧距 = -y - {half*1000:.0f}mm）")
    for nm in tip_names:
        rel = to_cube(sR["tips"][nm])
        d_front = -rel[1].item() - half
        print(f"     {nm:8s} rel=({rel[0]:+.4f},{rel[1]:+.4f},{rel[2]:+.4f})  -y面外侧距={d_front*1000:+.1f}mm")
    rel_w = to_cube(sR["wrist_p"])
    print(f"     wrist    rel=({rel_w[0]:+.4f},{rel_w[1]:+.4f},{rel_w[2]:+.4f})  world=({sR['wrist_p'][0]:+.3f},{sR['wrist_p'][1]:+.3f},{sR['wrist_p'][2]:+.3f})")
    print(f"     TCP      rel={tuple(round(v,4) for v in to_cube(sR['tcp']).tolist())}")
    print("     关节角：")
    for i, nm in enumerate(joint_names):
        print(f"        {nm:24s} {sR['jpos'][i].item():+.4f}")
    print("     指尖力(N)：" + "  ".join(f"{nm}={sR['forces'][k].item():.4f}" for k, nm in enumerate(tip_names)))

    # CSV 输出
    os.makedirs(args_cli.out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.abspath(os.path.join(args_cli.out_dir, f"scripted_grasp_v28_report_{ts}.csv"))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "name", "v0", "v1", "v2", "v3"])
        for nm in tip_names:
            t = sR["tips"][nm]
            w.writerow(["tip_world_m", nm, f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}", ""])
            r = to_cube(t)
            w.writerow(["tip_cube_m", nm, f"{r[0]:.6f}", f"{r[1]:.6f}", f"{r[2]:.6f}", ""])
        for i, nm in enumerate(joint_names):
            w.writerow(["joint_rad", nm, f"{sR['jpos'][i].item():.6f}", "", "", ""])
        for k, nm in enumerate(tip_names):
            w.writerow(["force_N", nm, f"{sR['forces'][k].item():.6f}", "", "", ""])
        w.writerow(["wrist_pos_m", "wrist_3_link", f"{sR['wrist_p'][0]:.6f}", f"{sR['wrist_p'][1]:.6f}", f"{sR['wrist_p'][2]:.6f}", ""])
        w.writerow(["wrist_quat", "wrist_3_link", f"{sR['wrist_q'][0]:.6f}", f"{sR['wrist_q'][1]:.6f}", f"{sR['wrist_q'][2]:.6f}", f"{sR['wrist_q'][3]:.6f}"])
        w.writerow(["cube_pos_m", "cube_obj", f"{cp[0]:.6f}", f"{cp[1]:.6f}", f"{cp[2]:.6f}", ""])
        for nm, dW, dC in cal_rows:
            for t in tip_names:
                w.writerow([f"handcal_dW_m.{nm}", t, f"{dW[t][0]:.6f}", f"{dW[t][1]:.6f}", f"{dW[t][2]:.6f}", ""])
                w.writerow([f"handcal_dC_m.{nm}", t, f"{dC[t][0]:.6f}", f"{dC[t][1]:.6f}", f"{dC[t][2]:.6f}", ""])

    print(f"\n[PR] 完成。CSV: {csv_path}")
    print("[PR] 把控制台全文（[PC7] 各表全段 + [PH] GUI 一眼（合围时两指各压 cube 哪个部位））发回——我据此定 v0.29（联合保持/首提）。")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
