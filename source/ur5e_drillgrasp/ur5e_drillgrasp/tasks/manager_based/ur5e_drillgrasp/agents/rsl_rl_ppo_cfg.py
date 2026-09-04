# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # v72: 16——用户缩短 rollout 加快每代迭代；Stage 1 纯接近任务动作简单，16 步够用。
    # v73: 16→24——用户要求。24 步覆盖更多轨迹(episode 299 步的 8%)，GAE/优势估计更准、
    # 更新更稳；代价是每代时间约 +50%。24×2048=49152，/4 个 mini-batch=12288，整除无余。
    num_steps_per_env = 16
    max_iterations = 10000
    # v70: 100→50——用户远程已改为每 50 轮保存（方便更细的续训/回退点，如 model_150）
    save_interval = 50
    # Stage 2: 从 Stage 1 最优 checkpoint 续训（18D + 触觉观测一致可 resume）
    #   ⚠️ 续训时命令行要 --experiment_name ur5e_grasp_stage1_v8（checkpoint 在那目录）
    experiment_name = "ur5e_grasp_stage1_v2"
    # ===== 非对称 actor-critic 观测组映射（借鉴 SoftHand）=====
    # actor  ← policy 组（109 维，维度不变）
    # critic ← policy + critic 特权组（109 + 14 = 123 维；特权项：
    #          cube 线/角速度 6、cube 绝对位置 3、指尖接触力 5，见 env_cfg.ObservationsCfg.CriticCfg）
    # 收益：success 依赖的"cube 稳定"和接触力 actor 看不到，critic 看得到
    #       → value 估计更准、优势函数质量更高、收敛更快更稳。
    # ⚠️ 兼容性警告：critic 输入 109→123，旧 checkpoint 直接 --resume 会因
    #   critic 权重形状不匹配（strict 加载）报错。旧 actor 权重仍可用——先用
    #   scripts/transfer_actor.py 迁移生成新 checkpoint，再从其续训。
    #   （play 旧 checkpoint 也会因同一原因失败，需先用旧版代码导出或先迁移。）
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "critic"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
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
        desired_kl=0.01,
        max_grad_norm=1.0,
    )