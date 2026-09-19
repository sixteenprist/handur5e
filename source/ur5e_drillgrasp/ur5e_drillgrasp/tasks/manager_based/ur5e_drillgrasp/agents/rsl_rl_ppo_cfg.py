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
    # v72: 16——用户缩短 rollout 加快每代迭代；Stage 1 纯接近任务动作简单，16 步够用。
    num_steps_per_env = 16
    max_iterations = 10000
    # v70: 100→50——用户远程已改为每 50 轮保存（方便更细的续训/回退点，如 model_150）
    save_interval = 50
    # [2026-09-14 四指共享] 从头训练：动作 18D→14D（手部四指共享 8D）。
    # [2026-09-15 力观测回归] fingertip_force 从 critic 移回 actor（98→103 维）：
    #   "始终贴面给力"需要策略感知接触力（闭环）；S2 未出成果 → 弃旧 ckpt 重训成本最低。
    #   旧 checkpoint（含 09-14/09-15 全部 run）一律不兼容，本版从头训练、不 resume。
    #   旧实验保留在 logs/rsl_rl/ur5e_grasp_4share_v1 等目录（勿覆盖）。
    experiment_name = "ur5e_grasp_4share_v1"
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
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,

        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        # [熵爆急救 2026-09-01] learning_rate 1e-3→5e-4：配合 fixed 降低单次更新幅度，加速熵回落。
        #   （v65 曾写 1e-3→5e-4，后被改回 1e-3，现恢复 5e-4）
        learning_rate=5.0e-4,
        # v15 修复: adaptive → fixed。adaptive 在策略稳定期(KL<0.005)会把 lr 一路 ×1.5 爬到
        # 1e-2 上限，随后一次大更新把策略推离稳定点 → value loss 爆炸(7.8)且不可逆退化(v15@572)。
        # fixed 保持 lr 恒定，杜绝 lr 漂移导致的突然失稳。
        # [熵爆急救 2026-09-01] adaptive→fixed：entropy 从 14 一路涨到 29 的元凶就是 adaptive 把 lr 漂移到 1e-2。
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        # desired_kl 仅 adaptive schedule 生效；当前 schedule="fixed"，此值为占位（无害）。
        #   可选实验：gamma 0.99→0.995（提起阶段的信用分配更远视，episode 300 控制步）。
        desired_kl=0.01,
        max_grad_norm=1.0,
    )