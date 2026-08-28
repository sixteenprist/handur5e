# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # v60: 32→64——从零训练 Stage 1 时 18D 动作空间探索期 std 膨胀(0.52)、reach/success 波动大。
    # episode 299 步，32 步只覆盖 10%→PPO 更新基于不完整轨迹，GAE 估计噪声大。
    # 64 让轨迹更长、价值/优势估计更准，从零学习更稳，缓解 std 膨胀与波动。
    # v65: 保持 64 不加长（96 会拖慢每代时间，用户要求）；稳定性靠 lr/entropy 保障
    num_steps_per_env = 16
    max_iterations = 5000
    # v70: 100→50——用户远程已改为每 50 轮保存（方便更细的续训/回退点，如 model_150）
    save_interval = 50
    # v65: v3→v5——v64 TCP offset (0.03,0.06,0.02) + 稳定性参数(lr 5e-4/entropy 0.003)；从零重训
    # v84: v5→v6——TCP 参考载体改 wrist_3_link + offset(0,0.10,0.08)（取反后红球在 w3 的 -y -z 方向）
    #   + v82 reward（reach sat_dist=0.06、palm_press -3.0、tcp_orientation -0.05）；从零重训
    # v92: v6→v7——TCP offset 改 (0,0.07,0.08)（掌心下 7-8cm→5cm）+ reward 重构：
    #   reach sat 1cm + 连续过渡（std 0.30）、success 阈值 1cm、tcp_too_close 改单调 z 深度
    #   （-3.0）、palm_press 关闭（base_link_1 误伤目标态）、bonus 1.5cm；从零重训
    experiment_name = "ur5e_grasp_stage1_v7"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
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
        # v65: 0.005→0.003——直接压制 entropy 快涨（崩溃前兆 entropy 变化很快）
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        # v65: 1e-3→5e-4——抑制单次更新幅度→reach 不再剧烈横跳；代价是收敛稍慢（平台期预计 600-700 轮）
        learning_rate=5.0e-4,
        # v15 修复: adaptive → fixed。adaptive 在策略稳定期(KL<0.005)会把 lr 一路 ×1.5 爬到
        # 1e-2 上限，随后一次大更新把策略推离稳定点 → value loss 爆炸(7.8)且不可逆退化(v15@572)。
        # fixed 保持 1e-3，杜绝 lr 漂移导致的突然失稳。
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )