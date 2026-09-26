# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# [2026-09-17 熵地板脚手架] 项目内 σ 护栏（ClampedActorCritic）注册——rsl_rl 用
#   eval(class_name) 在 runner 模块命名空间里找策略类，需项目侧把自定义类注册进去
#   （不碰 site-packages）。注册=挂名，不改变行为；启用/停用见 clamped_actor_critic.py 顶部常量。
from .clamped_actor_critic import register_clamped_actor_critic

register_clamped_actor_critic()


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # [2026-09-21 v6 归一化·用户决定] 32→24：回到官方任务标准（IsaacLab Franka Lift / Allegro
    #   in-hand 均为 24；rsl_rl/legged_gym 惯例 24）。此前 16/32 是针对性补丁，现配置已归一化。
    #   回退：32（更长信用分配窗口）/ 16（更快迭代）。
    num_steps_per_env = 24
    max_iterations = 10000
    # v70: 100→50——用户远程已改为每 50 轮保存（方便更细的续训/回退点，如 model_150）
    # [2026-09-21 v5.5] 50→25：峰值→退化窗口仅 ~30 iter（1018→1050），50 间隔漏掉峰值。
    # [2026-09-21 v5.6] 25→5：峰值窗口仅 ~10 iter（1006~1011），25 仍会漏掉；加密采峰。
    save_interval = 5
    # [2026-09-14 四指共享] 从头训练：动作 18D→14D（手部四指共享 8D）。
    # [2026-09-15 力观测回归] fingertip_force 从 critic 移回 actor（98→103 维）：
    #   "始终贴面给力"需要策略感知接触力（闭环）；S2 未出成果 → 弃旧 ckpt 重训成本最低。
    #   旧 checkpoint（含 09-14/09-15 全部 run）一律不兼容，本版从头训练、不 resume。

    # [2026-09-21 v5] 新流程：GUI 预置对置姿态（±0.005 轻随机）→ 只学"贴近+抓稳力封闭+提起"。
    #   从头训练（新初始位姿+精简奖励，旧 ckpt 不适用），新目录便于与 v4 对比。
    experiment_name = "ur5e_grasp_v5"
    # ===== 非对称 actor-critic 观测组映射（借鉴 SoftHand）=====
    # actor  ← policy 组（103 维 = 26 关节位置 + 26 关节速度 + 14 动作 + 3+3 相对向量
    #          + 4 cube 朝向 + 3+4 手掌位姿 + 15 指尖向量 + 5 指尖接触力；
    #          四指共享后 last_action 18→14；goal_pose 7 常量维已注释；
    #          指尖力 5 维 2026-09-15 从 critic 移回）
    # critic ← policy + critic 特权组（103 + 9 = 112 维；特权项：
    #          cube 线/角速度 6、cube 绝对位置 3，见 env_cfg.ObservationsCfg.CriticCfg）
    # 收益：success 依赖的"cube 稳定"actor 看不到，critic 看得到
    #       → value 估计更准、优势函数质量更高、收敛更快更稳。
    # ⚠️ 动作 14D / 观测 103D，与所有旧 checkpoint 不兼容——必须从头训练。
    #   （transfer_actor.py 只适用于"观测不变、仅 critic 维度变化"的迁移，本次不适用。）
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "critic"],
    }
    policy = RslRlPpoActorCriticCfg(
        # [2026-09-17 熵地板脚手架] 换为项目内子类（agents/clamped_actor_critic.py）：
        #   当前 σ 界值默认全关（-1）→ 行为与原生 "ActorCritic" 完全一致（先冒烟）。
        #   启用"手部维 σ 地板"：改 clamped_actor_critic.py 顶部 STD_FLOOR_HAND（建议先 0.10）。
        #   回退一行：class_name 改回 "ActorCritic"。
        class_name="ClampedActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # ═══ [2026-09-21 v6 PPO 归一化·用户决定] 对照调研（rsl_rl 默认 / IsaacLab Franka Lift /
    #   IsaacLab Allegro in-hand，见对话记录）把历史"防崩补丁"（epochs2/clip0.1/lr5e-5/
    #   entropy0.0015/fixed）全部回归标准值，采用官方灵巧手（Allegro）风格：
    #     lr 1e-4 + adaptive(desired_kl 0.01) / epochs 5 / minibatch 4 / clip 0.2 /
    #     entropy 0.002 / gamma 0.99 / lam 0.95 / max_grad_norm 1.0 / clipped value loss。
    #   说明：旧补丁是在"续训 lr 被优化器状态覆盖"bug 未发现时打的（train.py 已修复），
    #     且 σ 护栏 0.09/0.12 保留（用户设计）——算法其余部分不再有非标成分。
    #   ⚠️ 历史记录 adaptive 曾把 lr 漂到 1e-2 致崩（v15），但那是与 σ 自由等问题叠加；
    #     官方默认即 adaptive。监控重点新增 lr 曲线；若再见漂移，改 schedule="fixed" 即可。
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        # [2026-09-22 v8.2 监控自主] 5e-5→1e-5：进入"巩固期"——当前最佳策略已达
        #   持续对置 gate=1.0 + 41% 时段抬过 7.5cm；burst 循环边际递减。用极小 lr 长跑，
        #   让策略在好解附近缓慢精修（漂移速率 ∝ lr），配合 5 分钟监控+加密存档采峰。
        #   回退：5e-5 / 1e-4。
        # [2026-09-22 路径延长适配] 1e-5→5e-5：预设中指改伸展(距面 6.3cm)、需重新学"闭拢"，
        #   提高一档加速适应；稳定后回 1e-5。回退：1e-5。
        # [v16.1 保护 BC 解] 5e-5→1e-5：v16 实测——新盒面度量让 vloss 尖峰(31.7)期间
        #   PPO 把 BC 的盒距 1.7cm 侵蚀到 3.1cm。降 lr 让评论家先稳定、actor 慢移。
        #   回退：5e-5。
        # [v30f 稳后回档] 1e-5→3e-5：vloss 已回落(3.5~18)，恢复学习速度。
        #   回退：1e-5。
        learning_rate=3e-5,
        # [2026-09-21 v7.1 监控自主] adaptive→fixed：逐轮 lr 日志实测——adaptive 在
        #   1.0e-5 ↔ 1.7e-4 之间每轮横跳（KL 在 desired_kl/2 边界反复穿越），
        #   配合接触不连续把策略从峰值(iter~130, gate 1.0/力 0.68N 平衡/抬 2.7cm)缓慢推离。
        #   固定 lr=1e-4（官方 Franka Lift 同值）使更新平滑；回退：adaptive（若固定后学习停滞）。
        schedule="fixed",
        # [v13 Round-1·信用分配] 0.99/0.95 → 0.995/0.98：有效视野 ~100→200+ 步，
        #   覆盖"伸展→贴面"整段慢速闭合（Step0 审计后确认激励方向对、瓶颈是信用分配）。
        #   γ↑ 会抬高方差：若 vloss/回报剧烈震荡 → 回退 0.992/0.96；再回 0.99/0.95。
        gamma=0.995,
        lam=0.98,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )