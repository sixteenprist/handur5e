# UR5e + 16 自由度灵巧手抓取项目 · 代码详解（新人向）

> 目标：第一次接触本项目的人，按本文顺序阅读，可以对照源码把 `env_cfg` 与 `mdp/` 全部读懂。
> 版本基准：2026-10-19（`main @ 49c7fb6`）。文中路径均相对仓库根目录；行号不写死（会随版本变化），以"文件名 + 定义名"定位。
> 配套文档：`DEPLOY_INTERFACE.md`（真机接口）、`docs/PAPER_DRAFT.md`（论文草稿）、`CHECKPOINTS.md`（历史与结论索引）。

---

## 0. 阅读顺序（建议）

1. 本文 §2（IsaacLab Manager-based 环境是什么）→ 建立框架心智；
2. 本文 §3（`ur5e_drillgrasp_env_cfg.py`）→ 知道所有配置在哪；
3. 按需读 `mdp/` 各模块：观测（§4）→ 动作（§5）→ 奖励（§6）→ 课程（§7）→ 事件（§8）→ 终止（§9）→ 延迟（§10）；
4. 资产与训练器（§11–§12）→ 跑起来；
5. §14（一条完整调用链）与 §15（修改指南）→ 动手改。

---

## 1. 项目结构

```
ur5e_drillgrasp/
├── source/ur5e_drillgrasp/ur5e_drillgrasp/
│   ├── assets/
│   │   ├── robots/drill_ur5e.py        # UR5e+手的 USD 资产配置（执行器/传感器/初始位）
│   │   └── objects/drill.py            # 旧"电钻"物体配置（当前任务不用，留档）
│   └── tasks/manager_based/ur5e_drillgrasp/
│       ├── __init__.py                 # gym 任务注册：Template-Ur5e-Drillgrasp-v0
│       ├── ur5e_drillgrasp_env_cfg.py  # ★ 环境总配置（场景/动作/观测/奖励/终止/事件/课程/命令）
│       ├── ur5e_drillgrasp.py          # 环境类（几乎空壳，继承 ManagerBasedRLEnv）
│       ├── agents/
│       │   ├── rsl_rl_ppo_cfg.py       # PPO 超参 + runner 配置
│       │   └── clamped_actor_critic.py # σ 护栏（动作噪声标准差上下限）
│       └── mdp/                        # ★ 任务逻辑（manager-based 的"零件库"）
│           ├── actions.py              # 22 维动作：6 OSC + 16 手指
│           ├── observations.py         # 111 维观测项
│           ├── rewards.py              # 奖励函数（§0–§4 分区；§4 为归档死代码）
│           ├── curriculum.py           # 自动课程（信号 EMA + 10 个课程函数）
│           ├── events.py               # 每集重置逻辑（关节/远起/预备位）
│           ├── termination.py          # 终止条件
│           ├── object_geometry.py      # 形状抽象层（box/sphere；Phase 1）
│           ├── delay.py                # 延迟鲁棒化包装器（训练用）
│           └── commands/drill_pose_command.py  # 目标位姿命令（当前 config 挂载但观测未用）
├── scripts/rsl_rl/                     # train / play / eval_final / eval_deploy / cli_args
├── scripts/analysis/make_paper_figures.py  # 论文图生成
├── docs/                               # 论文草稿、图表、本文
├── logs/                               # 训练/评测日志与 checkpoint（gitignore）
└── DEPLOY_INTERFACE.md / CHECKPOINTS.md
```

包结构注意：Python 包路径是 `source/ur5e_drillgrasp/ur5e_drillgrasp/...`（外层 `source/<包名>` 是 setuptools 的 `package_dir`，第二个 `ur5e_drillgrasp` 才是真正的包）。导入时统一写 `from ur5e_drillgrasp.tasks.manager_based... import ...`。

---

## 2. 先修概念：IsaacLab 的 Manager-based 环境

任务**不是**写一个大 `step()` 函数，而是把职责拆给若干 **Manager**，每个 Manager 的条目都写在 *配置类*（`@configclass`）里：

| Manager | 配置类 | 作用 | 本项目的例子 |
|---|---|---|---|
| ActionManager | `ActionsCfg` | 把策略输出的"动作"翻译成关节目标/力矩 | `arm_action`, `hand_action` |
| ObservationManager | `ObservationsCfg` | 拼接观测向量（按组：policy / critic） | 10 个观测项 → 111 维 |
| RewardManager | `RewardsCfg` | 每步计算各项奖励并加权求和 | 19 项奖励 |
| TerminationManager | `TerminationsCfg` | 判断 episode 是否结束 | 6 个终止条件 |
| EventManager | `EventCfg` | 每集重置/启动时的状态改写 | 关节复位、远起、预备位、cube 复位 |
| CurriculumManager | `CurriculumCfg` | 训练中按需调整参数/奖励/事件 | 15 个课程项 |
| CommandManager | `CommandsCfg` | 采样目标命令（可选） | `drill_pose`（当前未用于观测） |

**一个策略步（30 Hz）大致流程**：

```
策略网络 a_t = π(o_t)
   │
   ▼
ActionManager：a_t → 关节位置目标/力矩（§5）
   │
   ▼
物理仿真：8 个 240 Hz 子步（decimation=8）
   │
   ▼
ObservationManager：读关节/物体状态 → o_{t+1}（§4）
RewardManager：算 19 项奖励加权和（§6）—— 其中"提起类"奖励由接触门控（§6.3）
TerminationManager：是否终止/超时（§9）
   │
   ▼
若 episode 结束：EventManager 重置该 env（§8）；CurriculumManager 可能调整参数（§7）
```

关键点：**配置即代码**。看到 `reward_term = RewTerm(func=mdp.xxx, weight=..., params=...)`，意思是"注册一个奖励项，名字叫 reward_term，用 `mdp/xxx.py` 里的函数计算，权重 weight"。

---

## 3. `ur5e_drillgrasp_env_cfg.py`（环境总配置）

文件自上而下依次定义：`SceneCfg → ActionsCfg → ObservationsCfg → EventCfg → RewardsCfg → CurriculumCfg → CommandsCfg → TerminationsCfg → apply_final_deploy_overrides() → Ur5eDrillgraspEnvCfg`。

### 3.1 SceneCfg（`Ur5eDrillgraspSceneCfg`）：场景里有什么

| 元素 | 内容 | 说明 |
|---|---|---|
| `num_envs / env_spacing` | 2048 / 4.0 m | 并行环境数与间距（观测里的世界坐标需减去 `env_origins`） |
| `replicate_physics` | True | 复制物理场景以加速 |
| `ground` | 100×100 平面 | 地面 |
| `table_collider` | 1.2×0.9×0.02 的盒子，z=0.74 | 桌面碰撞体（顶面 ≈0.75） |
| `cube_obj` | 6 cm（0.06 m）立方体，质量 0.05 kg，位置 (−0.14, 0.18, 0.78) | 抓取目标；`solver_position_iteration_count=16`（接触稳定） |
| `robot` | `DRILL_UR5E_CFG`（§11） | UR5e + 灵巧手 USD |

### 3.2 仿真参数与时间

- 物理步 `sim.dt = 1/240 s`，`decimation = 8` → **策略 30 Hz**；
- 每集 `episode_length_s = 10 s` → 300 个策略步；
- 摩擦：静/动都 **1.5**，restitution=0；
- PhysX：`enable_stabilization=True`，较大的接触/面片/栈显存配置（抓取接触多，防爆）。

### 3.3 ActionsCfg：动作怎么定义

**手臂（6 维）**：`SafeOperationalSpaceControllerActionCfg`
- 输出 = 任务空间增量 `[Δx,Δy,Δz (m), Δrx,Δry,Δrz (rad)]`，`position_scale = orientation_scale = 0.005`（即 raw=1 时每步 5 mm / 0.005 rad）；
- 底层 OSC：`target_types=["pose_rel"]`，重力补偿 + 惯性解耦；任务刚度 `(200,200,200,194,194,194)`，阻尼比 `(1.5,1.5,1.5,0.56,0.56,0.56)`；
- 安全限幅：`raw_clip=None, max_delta=None`（**当前关闭**；类默认 2.5/2.0，需要时打开）；
- TCP 参考点 = `wrist_3_link` 原点 + 局部偏移 `(0, 0.08, 0.11)`（与观测/奖励一致）。

**手（16 维）**：`GroupedHandActionCfg`（详见 §5）

| 参数 | 值 | 含义 |
|---|---|---|
| `four_mode` | `independent_j123` | 拇指 4 通道 + 食/中/无/小 各 (j1,j2,j3) |
| `scale` | 1.0 | 动作单位 = rad（目标 = 默认位 + 动作） |
| `clip_range` | 1.6 | raw 截断 ±1.6 |
| `max_delta` | 0.3 | 每步目标变化限幅（rad/步） |
| `j1_limit` | 0.10 | 四指 j1 侧摆增量限幅（rad） |
| `frozen_channels` | `(1,)` | 训练起始冻结拇指 t2；课程后段 `thaw→1` 放开 |
| `thaw` | 0.0 | 冻结通道放行系数（课程渐变） |

### 3.4 ObservationsCfg：观测向量（policy 111 维 + critic 9 维）

`PolicyCfg` 的 10 个 `ObsTerm`（按顺序拼接，`history_length=0`、不做 corrupt）：

| 顺序 | 名称 | 维 | 语义/单位 |
|---|---|---|---|
| 1 | `joint_pos` | 26 | q − q_default（rad） |
| 2 | `joint_vel` | 26 | rad/s |
| 3 | `last_action` | 22 | 上一步动作（动作空间值） |
| 4 | `tcp_to_cube` | 3 | cube 中心 − TCP（世界系，m） |
| 5 | `palm_to_cube` | 3 | cube − 掌心（世界系，m） |
| 6 | `cube_quat` | 4 | cube 朝向（wxyz） |
| 7 | `palm_pos` | 3 | 掌心位置 |
| 8 | `palm_quat` | 4 | 掌心朝向 |
| 9 | `fingertip_to_cube` | 15 | 5 指尖 ×（指尖→cube **表面**向量，已扣 3cm 半边长） |
| 10 | `fingertip_force` | 5 | 5 指尖净接触力**范数**（N） |

`CriticCfg` 在 policy 之外再拼 9 维特权量：`cube_lin_vel(3)`、`cube_ang_vel(3)`、`cube_pos(3)`（价值网络用真实状态，帮助更准）。

### 3.5 RewardsCfg：奖励项（19 项）

每项形如 `name: RewTerm = RewTerm(func=mdp.xxx, weight=w, params={...})`。当前终态/部署配置（`--final`）下权重如下（训练早期部分项不同，由课程调整）：

| 阶段 | 项 | 权重 | 作用一句话 |
|---|---|---|---|
| 门控 | `contact_gate` | 2.0 | 对向力质量（阈值 0.15N、k=1、软计数） |
| | `contact_align` | 1.0 | 接触力沿 cube ±y 的占比（力封闭方向） |
| | `contact_hold` | 2.0 | 接触占空比 EMA（治断触/粘抖） |
| S1 接近 | `tcp_reach` | 1.5→0（课程） | TCP 接近（终态归零） |
| | `tip_progress` | 12→0（课程） | 指尖闭拢进度（potential shaping，终态归零） |
| | `palm_cube_hold` | −1.0 | 防"掌心托举" |
| S2 四指 | `four_contact` | 0→1.5（课程） | 四指参与度软付款（拇指对置时每多一根 >0.15N） |
| | `four_pose_ref` | 0→1.0（课程） | 四指姿态软吸引（参考 (0,0.85,1.30)，σ0.6） |
| S2b 贴面 | `surface_proximity` | 2.0 | 指尖→盒面贴近（cfg：tanh 核 σ=0.10、盒面距离；五指尖等权） |
| | `finger_close` | 1.0 | 弯曲软门控（越近越值钱） |
| | `thumb_face_reach` | 3.0 | 拇指尖→对侧面中心的紧核吸引 |
| | `fingertip_press` | −0.2 | 防手指下压 cube |
| S3 提起 | `lift_height` | 15.0 | 净提升 8cm 饱和 × 提起门控（核心收益） |
| | `lift_success` | 5.0 | 净提升 ≥8cm 且真夹住 → 1 |
| | `lift_conform` | 1.5 | 提起期四指贴近/贴住 |
| | `lift_without_grip` | −1.5 | 无夹持却抬升（防抛起/托举） |
| 稳定 | `action_rate` | −0.15 | 动作变化率平方（封顶 10） |
| | `cube_motion` | 0→−0.8（课程） | cube 被推挤（速度/角速度） |
| | `cube_drift` | 0→−1.5（课程） | cube 水平漂移 |

> 具体公式见 §6；全局开关（`GATE_LIFT/FLEX_MIX/LIFT_FOUR`）见 §6.5。
> 参数注意：上表为末态值，例如 `contact_gate.thr` 训练起点为 0.03，由课程升到 0.15。

### 3.6 TerminationsCfg：终止条件（6 个）

| 项 | 函数 | 阈值/逻辑 |
|---|---|---|
| `time_out` | `mdp.time_out` | 满 300 步（框架 timeout） |
| `drill_drop` | `mdp.drill_dropped` | cube z < 0.0（掉落） |
| `abnormal_robot` | `mdp.abnormal_robot_state` | 任一关节速度 > 15 rad/s |
| `cube_out_of_bounds` | `mdp.cube_out_of_bounds` | cube z ∉ [0.5, 1.3] |
| `cube_drift_limit` | `mdp.cube_xy_drift_exceeded` | 相对本集起点水平漂移 > 4 cm（0.04 m） |
| `fingertip_overload` | `mdp.fingertip_force_overload` | 任一指尖力 > 1.5 N（深穿保护） |

成功**不会**终止 episode（继续给奖励/保持，避免"一提就结束"的价值偏差）。

### 3.7 EventCfg：每集重置（6 个事件）

| 事件 | 函数 | 作用 |
|---|---|---|
| `reset_robot` | `reset_joints_by_offset` | 全关节默认位 ±0.005 rad（速度 0） |
| `reset_middle_curriculum` | `reset_middle_curriculum` | 中指 j2/j3 固定 0.5236（30°）+±0.005（历史课程残留，当前与默认一致） |
| `reset_arm_far` | `reset_arm_far` | **远起**：在默认位基础上叠加 6 关节偏移 × `scale`（课程 0→1），±0.03 rad |
| `reset_thumb_curriculum` | `reset_thumb_curriculum` | 拇指 reset 课程（当前 `mix=0.0` → 实际不改写，保留机制） |
| `reset_cube` | `reset_root_state_uniform` | cube 复位到默认位（`pose_range={}` 精确；环境变量 `CUBE_POSE_RANGE` 可开随机化） |
| `reset_arm_prepose` | `reset_arm_prepose` | **易起预备位**：叠加另一组 6 关节偏移 × `pose_scale`（课程 1→0） |

远起/预备位的偏移量见 §8.2/§8.3。

### 3.8 CurriculumCfg：15 个课程项

| 课程项 | 函数 | 何时/做什么（关键参数） |
|---|---|---|
| `arm_far` | `curriculum_arm_far` | 远起 scale 阶梯 0→1（threshold 0.45，每档 ≥5000 步，只升不降） |
| `anneal_prepose` | `curriculum_param_anneal` | 预备位 `pose_scale` 1→0（4000 步） |
| `anneal_gate_thr` | `curriculum_param_anneal` | 接触门阈值 `contact_gate.thr` 0.03→0.15（6000 步） |
| `anneal_hand_preclose` | `curriculum_hand_preclose_anneal` | 手部预闭合整段前移（6000 步） |
| `anneal_indep_mix` | `curriculum_indep_mix_anneal` | 四指独立性 α 0→1（30000 步，16 维更慢） |
| `anneal_flex_mix` | `curriculum_flex_mix_anneal` | 四指弯曲聚合 min 权重 0→0.5（`after_term="indep_mix"`，12000 步） |
| `anneal_four_contact` | `curriculum_reward_anneal` | `four_contact` 0→1.5（6000 步） |
| `anneal_four_pose_ref` | `curriculum_reward_anneal` | `four_pose_ref` 0→1.0（`after_term="four_contact"`，6000 步） |
| `anneal_gate_lift` | `curriculum_gate_lift_anneal` | 提起门控 0.10/0.15→0.35/0.2 |
| `anneal_cube_motion` | `curriculum_reward_anneal` | `cube_motion` 0→−0.8 |
| `anneal_cube_drift` | `curriculum_reward_anneal` | `cube_drift` 0→−1.5 |
| `anneal_lift_four` | `curriculum_lift_four_anneal` | 四指贴近门 n 0→4（`after_term="four_pose_ref"`，12000 步） |
| `unfreeze_t2` | `curriculum_unfreeze` | 拇指 t2 软解冻（threshold 0.6，20000 步） |
| `anneal_tip_progress` | `curriculum_reward_anneal` | `tip_progress` 12→0（`after_term="t2_unfreeze"`，12000 步） |
| `anneal_tcp_reach` | `curriculum_reward_anneal` | `tcp_reach` 1.5→0（`after_term="tip_progress"`，12000 步） |

串行链：`indep_mix → flex_mix`；`four_contact → four_pose_ref → lift_four`；`t2_unfreeze → tip_progress → tcp_reach`；其余按各自 threshold 并行推进。所有课程共享同一成功率 EMA 信号（§7.1）。

### 3.9 CommandsCfg

`drill_pose = DrillPoseCommand`（`resampling_time_range=(1000,1000)`，`debug_vis=True`）。任务当前把目标固定为 cube 位置；**观测与奖励的现行实现并不读取该 command**（观测项都是相对 cube 真值/几何量）。保留它是为"电钻阶段"的目标约束扩展。

### 3.10 `apply_final_deploy_overrides(cfg)`：部署配置一键切换（非常重要）

训练配置里存的是**课程起始的"易值"**（预备位=1、预闭合、门控松、tip_progress=12、t2 冻结…）。而**部署/评测**必须用"课程末态"：

1. `curriculum = None`（冻结所有课程）；
2. `reset_arm_prepose.pose_scale = 0`（不再用易起预备位）；
3. `reset_arm_far.scale = 1.0`（远起 20 cm，标准起点）；
4. `hand_action.frozen_channels = ()`、`thaw = 1.0`（t2 自由）；
5. 全局门控/聚合到末态：`GATE_LIFT=0.35/0.2`、`INDEP_MIX.alpha=1.0`、`FLEX_MIX.min_w=0.5`、`LIFT_FOUR.n=4.0`；
6. 奖励末态：`contact_gate.thr=0.15`、`four_contact.weight=1.5`、`four_pose_ref.weight=1.0`、`tip_progress.weight=0`、`tcp_reach.weight=0`、`cube_motion=-0.8`、`cube_drift=-1.5`；
7. 环境变量：
   - `CUBE_POSE_RANGE=<半幅m>`：每集 cube 在 x/y ±半幅均匀随机（默认 0=关）；
   - `ABLATE="term1,term2"`：把对应奖励权重置 0（消融实验用）；
   - `INDEP_MIX0=<alpha>`：模块导入时设定四指独立性 α 的初值（续训保值，不置则从 0 起）。

**为什么必须用**：直接用原始 cfg 推理会出现"观测/动作/起点分布错配"，成功率骤降（历史实测）。`play.py / eval_final.py / eval_deploy.py` 都通过 `--final` 或直接调用本函数来保证一致。

### 3.12 物体形状抽象（Phase 1：box / sphere）

`mdp/object_geometry.py` 把"写死的立方体几何"集中为可替换接口，用**环境变量**设定（进程级）：

| 变量 | 取值 | 说明 |
|---|---|---|
| `OBJECT_KIND` | `box`（默认）/`sphere` | 物体形状 |
| `OBJECT_SIZE` | 0.03（默认） | box=半边长；sphere=半径（m） |

- **box（默认）**：spawn、初始高度与全部公式**逐位不变**（已用固定种子观测快照 + v58 100 集验收验证）；
- **sphere**：`SceneCfg` 的 spawn 换成球（质量 0.05kg），球心高度 = 0.75 + 半径；观测/奖励经
  `object_geometry.surface_distance/surface_vector` 自动走球面公式；`thumb_face_reach` 与
  `contact_alignment` 的对置轴改为"拇指→参考指尖"的水平方向（方块仍用 cube 局部 ±y）；
- **限制**：当前一次进程只支持一种形状（便于分别训练/评测）。"一个权重同时抓多种形状"
  需要"每环境形状 + 物体池"机制（Phase 2，尚未实现）。

### 3.11 EnvCfg 与任务注册

`Ur5eDrillgraspEnvCfg` 只是把上面所有 Cfg 组合起来，并设 `decimation=8`、`episode_length_s=10`、`sim.dt=1/240`。
`tasks/.../ur5e_drillgrasp/__init__.py` 用 `gym.register` 注册任务 id **`Template-Ur5e-Drillgrasp-v0`**（入口 `ManagerBasedRLEnv` + 本 cfg + PPO cfg）。

---

## 4. `mdp/observations.py`（观测项实现）

该文件实现观测项与内部辅助函数（本任务配置用到其中 10 项），全部返回 `(num_envs, D)` 张量。重要常量：

| 常量 | 值 | 用途 |
|---|---|---|
| `ACTIVE_TIP_NAMES` | `("thumb4","index4","middle4","ring4","little4")` | 指尖顺序（观测/奖励共用） |
| `ACTIVE_FOUR_INDICES` | `(1,2,3,4)` | 四指在 5 指尖中的下标 |
| `_CUBE_HALF_SIZE` | 0.03 m | cube 半边长（指尖到表面向量要扣） |
| `_BODY_OFFSET` | `(0.0, 0.08, 0.11)` | TCP 相对 `wrist_3_link` 的局部偏移 |

| 函数 | 计算方式 | 备注 |
|---|---|---|
| `object_position` / `object_orientation` / `object_lin_vel` / `object_ang_vel` | 直接读 `cube_obj` 世界位姿/速度（位置/四元数/线速度/角速度） | critic 用 |
| `palm_position` / `palm_orientation` | `base_link_1` 的 link 位置/朝向（四元数） | 世界系 |
| `tcp_position` | `wrist_3` 位置 + R·offset | 与奖励 `_get_tcp_pos` 同源 |
| `tcp_to_cube` / `palm_to_cube` | cube 中心 − TCP/掌心 | 相对向量 |
| `fingertip_to_cube_surface` | 每指尖：`(cube−tip)·clamp((d−0.03)/d,0)`，5×3 展平 15 维 | 已扣半边长；无接触时≈实际到表面距离 |
| `fingertip_contact_force` | 5 个 contact sensor 的合力范数 | 单位 N；无接触=0 |
| `joint_pos_rel` / `joint_vel_rel` | q−默认 / 速度 | 26 维（手 20 + 臂 6） |
| `last_action` | `env.action_manager.action` | 22 维，动作空间值 |

内部工具：`_get_body_id` 用 `(id(robot), name)` 做缓存；`_robot_body_position` 读 body link 世界位置；`_palm_id` / `_tcp_id` 分别为掌心 `base_link_1` / `wrist_3_link` 的 body id；`generated_commands` 是命令类观测的辅助函数（当前观测配置未挂载）。

> 真机映射：`cube_*` 来自相机位姿估计；`palm/fingertip/tcp` 来自关节编码器 + 手部 FK；`fingertip_force` 来自实机触觉传感器。详见 `DEPLOY_INTERFACE.md`。

---

## 5. `mdp/actions.py`（动作实现）

### 5.1 `GroupedHandAction`（16 维手指）

**通道布局**（`four_mode="independent_j123"`）：

```
ch0-3  : thumb1, thumb2, thumb3, thumb4
ch4-6  : index1, index2, index3
ch7-9  : middle1, middle2, middle3
ch10-12: ring1, ring2, ring3
ch13-15: little1, little2, little3
```

**每步处理链**（顺序很重要）：

1. `raw = 策略输出`；可选 `bias` 叠加；`clip_range` 截断到 ±1.6；
2. ×`scale`(1.0) → 弧度增量；
3. **冻结通道**：`raw[frozen] *= thaw`（thaw=0 完全冻结，课程 0→1 渐变放开）；
4. **拇指 ch0/ch1 符号翻转**（策略正值对应 target 减小，符合物理弯曲方向）；
5. 组装到整手关节向量（j4 不写 → 保持默认 0°）；
6. 四指 **j1 clamp ±0.10**（防侧摆互挤）；j2/j3 不限；
7. **绝对位置目标 = `default_joint_pos + 动作`**；
8. **步间限幅**：`target` 相对上一步最多 ±`max_delta`(0.3 rad)；
9. `set_joint_position_target(target)`。

**独立性课程 α**（`INDEP_MIX` 全局字典；初值可由环境变量 `INDEP_MIX0` 指定，续训保值）：当 α<1 时，四指每指动作向"食指通道"混合：`a_i ← α·a_i + (1−α)·a_index`。训练早期四指同动（易探索），后期各指独立。**推理时必须把 α 置 1**（`apply_final_deploy_overrides` 已处理），否则行为=共享模式。

**其它模式**（历史/回退）：`"middle"`（仅中指）、`"shared"`（四指共享 8 维）、`"independent_j23"`（12 维）。切换 `four_mode` 会改变动作维度 → 必须重训（旧权重维度不符）。

### 5.2 `SafeOperationalSpaceControllerAction`（6 维手臂）

在 IsaacLab 原生 OSC 动作外再包一层防护：

- `raw_clip`：限制 raw 绝对值（默认 2.5，当前 cfg 关闭=None）；
- `max_delta`：限制 raw 步间变化（默认 2.0，当前关闭）；
- 顺序：先幅度、后步间；`_last_raw` 记录实际生效值（防 windup）；episode 重置时清零。

作用：训练崩坏/σ 爆时防止手臂"乱飞乱晃"（`σ` 护栏是另一层，见 §12）。

---

## 6. `mdp/rewards.py`（奖励实现）

文件按 §0–§4 组织（2026-10-15 重排）：

- **§0 全局开关/常量**：`_FINGER_JOINT_PATTERNS`、`_FINGER_JOINT_WEIGHTS`、`GATE_LIFT`、`FLEX_MIX`、`LIFT_FOUR` 等；
- **§1 基础工具**：位置/距离/缓存/分裂/弯曲/力向量；
- **§2 接触与门控原语**：`_good_contact` 家族；
- **§3 现役奖励**（按 S1/S2/S2b/门控消费/S3/稳定/休眠 排列）；
- **§4 归档区**：历史死代码（全注释；恢复=去掉行首 `# `）。

### 6.1 基础工具（§1）

| 函数 | 作用 |
|---|---|
| `_get_cube_pos` / `_get_tcp_pos` / `_tcp_cube_dist` | cube 质心 / TCP 世界位置 / 距离 |
| `_get_fingertip_pos(name)` | 5 指尖中某个 link 的世界位置 |
| `_tip_cube_box_dist(name)` | 指尖到 cube **精确盒面距离**：`‖clamp(|p−c|−0.03, min=0)‖` |
| `_finger_flex` / `_finger_flex_per_finger` | 关节角加权弯曲度（0=伸直，~1.57=全弯） |
| `_thumb_four_split(per)` | 把每指量拆成"拇指组 + 四指组"（默认四指均值；`four_idx=(2,)` 可只取中指） |
| `_fingers_side_hysteresis` | 判断四指在 cube 的哪一侧（±1，带滞回防抖） |
| `_get_finger_joint_ids` | 按名称模式取每根手指的全部关节 id（缓存） |
| `_fingertip_cube_center_dists` | 五指尖→cube 质心的纯欧氏距离（不扣半边长；旧版 finger_reaching 专用） |
| `_finger_j123_ids` | 四指各自 (j1,j2,j3) 关节下标（缓存；动作/姿态参考用） |
| `_fingertip_force_vectors` | 5×3 世界系接触力向量 |

### 6.2 门控原语（§2）——**奖励设计的核心**

1. **`_good_contact(thr=0.15, k=1, soft=0.05)`：接触质量 q(0~1)**
   - 每根手指：`sigmoid((力 − thr)/soft)`（力越大越接近 1）；
   - 计数门：`sigmoid((count − k)/0.5)`（至少 k 根手指有接触）；
   - 质量 = `拇指门 × 四指计数门`。
2. **`_lift_gate(deadzone, scale, ema_alpha=0.1)`：提起门控**
   - 对 q 做 EMA（防"瞬时一碰"），减去死区后 `tanh` 归一；
   - × `_contact_alignment`（力必须沿夹持轴，纯竖直托举≈0）；
   - × `_four_near_gate`（四指贴近门，见下）；
   - **没有持续、方向正确、四指贴近的夹持 → 提起类奖励≈0**。
3. **`_four_near_gate(ema_alpha=0.1)`：四指贴近门**
   - 每指贴近度 `1 − tanh(盒距/σ)`（σ=0.05）；
   - 软计数 = 各指贴近度求和，过 `n−0.5` 的 sigmoid（n 由课程 0→4）；
   - 只要求"贴近"，不要求接触力；`LIFT_FOUR["touch"]` 为休眠的"轻触因子"（默认 0=关）。
4. **`_contact_alignment(force_thresh=0.03)`**：接触力沿 cube 局部 y 轴（夹持轴）的占比。
5. **`_contact_gate(scale, thr, k, soft)`**：`_good_contact` 的薄封装（保留旧签名，`scale` 参数已不使用）。
6. **`_lift_gain(max_height)`**：净提升比例 `clamp(z−z_start, 0, max_height)/max_height`（提起两兄弟项的公共部分）。
7. **`_gate_k` / `_gate_lift_params`**：从 cfg/全局读 k 与 GATE_LIFT 参数（单一来源，课程可调）。

### 6.3 现役奖励（§3，按阶段）

| 阶段 | 函数 | 公式/要点 |
|---|---|---|
| S1 | `tcp_reach_reward(σ=0.10)` | `1 − tanh(d/σ)`（长尾接近；课程退火到 0） |
| S1 | `tip_progress_reward(tip_names, metric)` | 逐指尖"本步距离 − 上步距离"求和（potential shaping；课程退火到 0） |
| S1 | `palm_cube_rel_pose_penalty(...)` | 掌心相对 cube 的位置/姿态漂移软惩罚（防悬停抖动） |
| S2 | `four_contact_reward(thr=0.15, soft=0.05)` | 拇指已对置接触时，四指每多一根超阈 → 多付 1/4 |
| S2 | `four_pose_ref_reward(ref=(0,0.85,1.30), σ=0.6)` | 四指 (j1,j2,j3) 向参考的手型吸引（治"折死大钩"） |
| S2b | `surface_proximity_reward(margin=0.03, kernel, sigma, metric)` | 指尖→盒面贴近；cfg 实际 `kernel="tanh"`、`sigma=0.10`、`metric="box"`（线性核 `clamp((margin−d)/margin,0,1)` 为默认/备用），五指尖等权 |
| S2b | `finger_close_reward(gate_std, ...)` | 弯曲度 × TCP 距离软门（越近越值钱） |
| S2b | `thumb_face_reach_reward(σ, ...)` | 拇指尖→对侧面中心的 `1−tanh(d/σ)`（最后 1cm 引导） |
| S2b | `fingertip_press_penalty(...)` | 指尖合力向下分量（防把 cube 压桌） |
| 门控消费 | `contact_gate_reward(...)` | 直接返回 `_good_contact`（即时"抓住"付款） |
| 门控消费 | `contact_alignment_reward(...)` | 力沿夹持轴的占比（防斜擦） |
| 门控消费 | `contact_persistence_reward(...)` | 接触占空比 EMA（治断触/颤） |
| S3 | `lift_height_reward(max=0.08, scale=0.2)` | `净提升/8cm`（饱和）× `_lift_gate` |
| S3 | `lift_success_reward(max=0.08, ...)` | 净提升≥8cm 的 0/1 指示 × `_lift_gate`（0~1，持续给） |
| S3 | `lift_conform_reward(...)` | 提起期四指尖贴近/贴住的一致性 |
| S3 | `lift_without_grip_penalty(...)` | 抬升但没有夹持质量（防抛起） |
| 稳定 | `cube_motion_penalty(v_std, w_std)` | `tanh(|v_xy|/0.05)+tanh(|ω|/1.0)` |
| 稳定 | `cube_drift_penalty(...)` | `tanh(|xy−起点|/0.02)` |
| 稳定 | `action_rate_penalty()` | `Σ(Δa)²`（clamp max=10） |
| 休眠 | `clamp_cube_xy()` | 锁定 cube xy/朝向（cfg 中已注释；恢复可解锁回退） |

### 6.4 为什么这样设计（读代码时的"意图"）

- **接触门控**是奖励链的"电源开关"：位置类奖励负责"引导"，力类门控负责"不许取巧"；`lift_*` 的所有收益都必须经过 `_lift_gate`。
- **四指贴近门**把"四指要贴面"从"软愿望"变成"提取得分的前提"（n=4 时四指都近才拿到完整提起奖励）。
- 奖励项之间的**权重/退火**由课程控制（§3.8），因此同一函数在不同训练阶段权重不同是正常的。

### 6.5 全局开关（模块级字典，课程直接改）

```python
GATE_LIFT = {"deadzone": 0.10, "scale": 0.15}   # 提起门控起点；课程→0.35/0.2
FLEX_MIX  = {"min_w": 0.0}                       # 四指弯曲聚合 min 权重；课程→0.5
LIFT_FOUR = {"n": 0.0, "sigma": 0.05, "touch": 0.0, "touch_soft": 0.02}  # 贴近门；n 0→4
```
`INDEP_MIX`（在 `actions.py`）同理：`{"alpha": 0.0}` → 1.0。

**推理/评测必须在同一进程里把这些全局量置到末值**（`apply_final_deploy_overrides` 已做），否则动作/奖励行为与训练分布不一致。

### 6.6 归档区（§4）

29 段历史死代码（旧奖励/旧逻辑）以注释形式集中存放，每段带 `# ===== [归档 ...] =====` 标记。不要直接启用；如需恢复，从 git 历史取原始实现更安全。

---

## 7. `mdp/curriculum.py`（自动课程）

### 7.1 信号与通用机制

- **信号**：`_signal_ema` 从 `RewardManager` 的 `lift_success` 项（权重归一化后 0~1）取均值，再做 EMA。**所有课程项共享同一 EMA 状态**（`env._cur_ema`），每步会被约 15 个课程项各调一次 → 实际等效 α≈1.5%/步（见函数注释的 2026-10-15 校正）。
- **判定**：`_tick(check_every)` 按环境步计数（幂等）；满足"success EMA ≥ threshold **且** 连续 `require_k` 次（间隔 check_every 步）"才推进。
- **进度**：`num_steps` 内线性渐变；若 `pause_below` 且 EMA 跌破阈值 → **暂停（不倒退）**；恢复后继续。
- **门控**：`gate_min_scale` 控制课程项的"参与度"（0=暂不生效，1=完全生效），用于"先升到某再把参数交给课程"的串联。
- **串行**：`after_term="<term_name>"` 表示必须等该课程项完成才启动。

### 7.2 课程函数：通用（3 个）+ 专用（6 个）

| 函数 | 改什么 | 典型用于 |
|---|---|---|
| `curriculum_reward_anneal` | 奖励**权重**（当前值 → target_weight） | 四指参与/姿态、cube 惩罚、tip/tcp 退火 |
| `curriculum_param_anneal` | 奖励**参数**（key → target_value） | 门控阈值 `contact_gate.thr`、预备位 `pose_scale` |
| `curriculum_unfreeze` | 动作 `thaw`（冻结通道放行系数） | 拇指 t2 软解冻 |
| `curriculum_gate_lift_anneal` | `GATE_LIFT`（deadzone/scale） | 提起门控收紧 |
| `curriculum_hand_preclose_anneal` | 手部默认关节位（预闭合） | 让手指"一开始就半弯" |
| `curriculum_flex_mix_anneal` | `FLEX_MIX.min_w` | 四指聚合从均值→半 min |
| `curriculum_indep_mix_anneal` | `INDEP_MIX.alpha` | 四指从共享→独立 |
| `curriculum_lift_four_anneal` | `LIFT_FOUR.n` | 四指贴近门 0→3→4 |
| `curriculum_arm_far` | 远起事件 `scale`（阶梯） | 起点从近到远 |

实现细节：`_get_event_cfg` 从 `EventManager` 找到事件项并直接改 `cfg.params`（因为事件每次调用都从 cfg 读参数）；奖励权重直接改 `RewardManager._term_cfgs[i].weight`。

### 7.3 历史坑（读代码会看到大段注释）

- **暂停计数回滚缺陷**：旧实现 `held += (cnt−t0−held)` 在首次暂停时把已累计进度清零 → 退火倒退（历次崩坏隐形主因）；现在改为记录 `_last`、只累计增量。
- **并发退火叠加会崩**（v52 教训）：α 与 flex 必须串行（`after_term`）。
- **一步到位会崩**：贴近门 n=4 直上会让 lift 骤降；先 n=3 再 n=4。

---

## 8. `mdp/events.py`（重置事件）

### 8.1 `reset_middle_curriculum`
把中指 j2/j3 覆写为固定值（当前 cfg 下 `mix_ref=-1, j2=j3=0.5236`，与资产默认一致）± noise。历史用途：让"出生时中指有弯曲"，便于从零学习闭拢；现在作为兼容保留。

### 8.2 `reset_arm_far`（远起）
在 `reset_robot` 之后，把 6 个手臂关节偏移 `scale × Δ`：

| 关节 | Δ (rad) |
|---|---|
| shoulder_pan | +0.0148 |
| shoulder_lift | +0.2260 |
| elbow | +0.1196 |
| wrist_1 | −0.3612 |
| wrist_2 | +0.0182 |
| wrist_3 | −0.0564 |

`scale=1` 时 TCP 到 cube 正上方约 20 cm（xy 不变）；±0.03 rad 噪声。**课程从 0 阶梯升到 1**。

### 8.3 `reset_arm_prepose`（易起预备位）
另一组 6 关节偏移（`pose_scale` 缩放）：

| 关节 | Δ (rad) |
|---|---|
| shoulder_pan | −0.0918 |
| shoulder_lift | −0.1775 |
| elbow | +0.0558 |
| wrist_1 | −0.0296 |
| wrist_2 | −0.2253 |
| wrist_3 | +0.0349 |

作用：课程早期起点更接近旧版"低悬停 + 腕预备位"（已验证可学）；`pose_scale→0` 后起点变为"高处下落"，由策略适应。

### 8.4 `reset_thumb_curriculum`
按 `mix` 比例把拇指预置到成功姿态（-0.70,-1.24,0.79,0.70）±0.05。当前 `mix=0` → 不改写（机制保留）。

### 8.5 cube 复位
用 IsaacLab 的 `reset_root_state_uniform`，默认 `pose_range={}` → cube 每集精确回默认位。**`CUBE_POSE_RANGE` 环境变量**在 `apply_final_deploy_overrides` 里注入随机化（±半幅，x/y）。

---

## 9. `mdp/termination.py`（终止条件）

| 函数 | 判定 | 设计意图 |
|---|---|---|
| `time_out` | 步数 ≥ max−1 | 正常超时 |
| `drill_dropped` | cube z < 0 | 掉落 |
| `abnormal_robot_state` | 任一关节速度 > 阈值 | 物理爆炸/求解异常（阈值 cfg=15） |
| `cube_out_of_bounds` | z ∉ [0.5,1.3] | 被顶飞/弹出 → 隔离极端样本 |
| `cube_xy_drift_exceeded` | 相对本集起点漂移 > 4cm（每集重置时缓存起点） | 防"拖动 cube 蹭提"退化解 |
| `fingertip_force_overload` | 任一指尖力 > 阈值（cfg=1.5N） | 深穿保护（正常接触 <1N） |

另有归档函数 `cube_launched`（不在 cfg 中启用）。

---

## 10. `mdp/delay.py`（延迟鲁棒化，训练用）

`DelayVecEnvWrapper` 包在 `RslRlVecEnvWrapper` 外面，**只影响训练**：

- **观测延迟 D_obs**：策略收到 `t−D_obs` 步的 policy 观测（critic 不延迟）；
- **动作延迟 D_act**：t 步真正下发的是 `t−D_act` 步的动作（环境 `last_action` 随之"迟到"，与真机一致）；
- **采样**：每个环境在每集重置时独立采样 `D_obs∈{0..obs_max}`、`D_act∈{0..act_max}`，集内不变；
- **缓冲**：观测/动作各一个环形缓冲（长度=最大延迟+1）；重置环境时把缓冲填成新集首帧/零动作，防跨集泄漏；
- **开关**：环境变量 `DELAY_OBS_MAX` / `DELAY_ACT_MAX`（train.py 注入；默认都不设=完全透传，行为与未启用一致）；
- **调试字段**：`debug_raw_policy`、`debug_applied`、`debug_flushes`（探针用）。

训练配方（v58）：`DELAY_OBS_MAX=1 DELAY_ACT_MAX=1` + v57 续训 2000 步。**如果混合训练的最差延迟格子成功率为 0%（如之前的 2/1），回报方差会导致训练崩坏**——先做集成校验，再决定档位。

---

## 11. `assets/robots/drill_ur5e.py`（资产）

- **USD**：`/home/jiangli/zhaoyucheng/assets/drill_ur5e_v1.usd`（绝对路径；换机需改）；`activate_contact_sensors=True`；
- **初始位**（= `default_joint_pos`，观测/动作的基准）：手臂 6 关节（完整默认位表见 `DEPLOY_INTERFACE.md` §4）+ 手：thumb1 −0.698 / thumb2 −1.396 / thumb3 0.873 / thumb4 0.698；食 j1 −0.087、四指 j2=j3=0.524、j4=0；
- **执行器**：
  - 手臂：`stiffness=0.0, damping=0.0`（纯力矩，OSC 直通），`effort_limit_sim=300.0`；
  - 手：`ImplicitActuatorCfg`，`stiffness=35.0, damping=8.0, effort_limit_sim=3.0`（Nm）（位置目标下的隐含 PD；effort 上限防深穿求解爆炸）；
- **自碰撞**：`enabled_self_collisions=False`。

`assets/objects/drill.py` 是旧"电钻"配置（当前任务不 spawn，留档）。

---

## 12. 训练器与 σ 护栏

- **PPO 配置**（`agents/rsl_rl_ppo_cfg.py` 的 `PPORunnerCfg`）：`num_steps_per_env=24`、`max_iterations=10000`（实际运行常用 `--max_iterations` 覆盖）、`save_interval=10`；网络 512-256-128 ELU（actor/critic 同构）；`learning_rate=3e-5`、`gamma=0.995`、`lam=0.98`、`clip_param=0.2`、`entropy_coef=0.002`、`num_learning_epochs=5`、`num_mini_batches=4`、`desired_kl=0.01`、`max_grad_norm=1.0`；观测归一化开启。
- **σ 护栏**（`agents/clamped_actor_critic.py` 的 `ClampedActorCritic`；由 `register_clamped_actor_critic()` 注册、`_build_bounds()` 生成上下限）：把动作噪声标准差按组夹在 `[floor, cap]`：臂 0.09、手 0.12（**下限=上限=固定**）→ 保证最小探索预算、防熵坍缩。参数是模块级常量（`STD_FLOOR_ARM/HAND`、`STD_CAP_ARM/HAND`，`-1`=关）。
- **训练入口**（`scripts/rsl_rl/train.py`）：`--final` 调用 `apply_final_deploy_overrides`；延迟注入由环境变量触发；resume 时优化器学习率按配置覆盖（见文件内补丁）。

---

## 13. 脚本速览

| 脚本 | 用途 | 关键点 |
|---|---|---|
| `train.py` | 训练 | `--final`（部署分布微调）、`--resume --load_run --checkpoint`、`--max_iterations`（续训时是"再加多少"） |
| `play.py` | GUI 回放 | 必须 `--final`（否则分布错配）；`--num_envs 16` |
| `eval_final.py` | 100 集确定性验收 | 成功=净提升 ≥8cm；输出提起/漂移/力 |
| `eval_deploy.py` | 17 场景部署裕度 | 动作低通/速率/观测延迟/动作延迟/噪声/组合；输出成功率/手颤 |
| `make_paper_figures.py` | 论文图 | 从 TB 拼接 v57 曲线；消融曲线从文本日志解析 |

---

## 14. 一条完整调用链（以训练一步为例）

```
train.py
  └─ gym.make("Template-Ur5e-Drillgrasp-v0", cfg=Ur5eDrillgraspEnvCfg())
       ├─ SceneCfg 建场景（2048 env，并行 env 间距 4m）
       ├─ EventManager：reset_robot → reset_middle_curriculum → reset_arm_far → reset_thumb_curriculum
       │                 → reset_cube → reset_arm_prepose（按 EventCfg 顺序）
       ├─ ObservationManager：policy 组 10 项 → 111 维
       └─ RslRlVecEnvWrapper（+ DelayVecEnvWrapper 可选）
  └─ PPO 循环：
        a = policy(obs)                      # 22 维
        obs, r, done, info = env.step(a)
            ├─ ActionManager：arm OSC（×0.005）/ hand 目标（default+a，clamp/限幅）
            ├─ 物理 8 子步（240Hz × 8）
            ├─ RewardManager：19 项加权和（提起类经 _lift_gate）
            ├─ TerminationManager：6 项
            └─ 若 done：EventManager 重置 + CurriculumManager 推进
```

评测同理，只是 `env.reset()` 时的课程被关闭（`--final`），动作取策略均值。

---

## 15. 修改指南 / 常见坑

**加一个新的奖励项**
1. 在 `rewards.py` §3 对应阶段写函数（输入 `env`，返回 `(N,)`）；
2. 在 `env_cfg.RewardsCfg` 加 `name: RewTerm = RewTerm(func=mdp.xxx, weight=..., params=...)`；
3. 若需要课程：加 `CurrTerm(func=curriculum_reward_anneal, params={"term_name": "name", ...})`，并设 `after_term` 串行关系。

**改课程**
- 先看 §7.1 的机制；新课程项一律用它提供的 `_tick/_signal_ema` 模式；
- 切勿并发叠加多个退火（v52 崩坏教训）；用 `after_term` 串行。

**改动作空间**
- 改 `four_mode` → 动作维度变 → **必须从头训练**；旧 ckpt 不兼容；
- 改 `scale/j1_limit/max_delta` 会影响行为与可学性（改后建议从头或长微调）。

**改观测/奖励后评测**
- 一定用 `--final`（或调用 `apply_final_deploy_overrides`），并保证 `INDEP_MIX/FLEX_MIX/GATE_LIFT/LIFT_FOUR` 全局量与训练末态一致；
- 评测与训练必须同进程设置全局量（Python 模块级字典不会自动同步）。

**延迟随机化**
- 先做"真实策略 × 固定延迟"的集成校验；最差格子成功率为 0% 的混合不要训（会崩）；档位从温和开始。

**常见报错**
- 成功率骤降：忘了 `--final`；或全局量没同步；
- 维度不符：换了 `four_mode`/观测项但加载旧权重；
- 训练崩坏：先看"集长/价值损失"曲线；检查是否并发退火/最差格子不可学/约束过紧。

---

## 16. 术语表

| 词 | 含义 |
|---|---|
| OSC | 操作空间控制（任务空间阻抗 + 重力补偿，底层出力矩） |
| GATE_LIFT | 提起门控参数（deadzone/scale），课程渐变 |
| LIFT_FOUR | 四指贴近门（n=要求贴近的指数，σ=贴近核宽度） |
| FLEX_MIX | 四指弯曲聚合权重（min/mean 混合） |
| INDEP_MIX | 四指独立性 α（0=共享食指通道，1=各指独立） |
| INDEP_MIX0 | 环境变量：模块导入时 α 的初值（续训保值） |
| thaw | 冻结动作通道的放行系数（0=冻结，1=放开） |
| after_term | 课程串行依赖（前一项完成后才推进） |
| gate_min_scale | 课程项参与度（0=暂不生效，1=完全生效） |
| CUBE_POSE_RANGE | 评测/训练时 cube 位置随机化半幅（m） |
| DELAY_OBS_MAX / DELAY_ACT_MAX | 训练时观测/动作最大延迟步数 |
| ABLATE | 消融开关（逗号分隔的奖励项名 → 权重置 0） |
| OBJECT_KIND / OBJECT_SIZE | 物体形状与尺寸（Phase 1：box/sphere，进程级） |

---

_写作基准：`main @ 49c7fb6`（2026-10-19）。若代码演进后与本文不符，以代码为准并同步更新本文件（尤其是 §3 的配置表与 §6/§7 的列表）。_
