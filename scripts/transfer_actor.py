#!/usr/bin/env python3
"""
Actor-Only Transfer: 从 Stage 1 checkpoint 提取 Actor 权重，注入 Stage 2 新模型。
Critic 和优化器从零初始化。

用法:
  python scripts/transfer_actor.py \
    --source logs/rsl_rl/ur5e_grasp_stage1_v2/2026-07-24_15-10-35/model_5000.pt \
    --output logs/rsl_rl/ur5e_grasp_stage2/stage2_init.pt

然后从 output 续训:
  python scripts/rsl_rl/train.py --task Template-Ur5e-Drillgrasp-v0 \
    --num_envs 1024 --headless \
    --resume True --load_run <folder> --checkpoint stage2_init.pt
"""

import argparse
import torch
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Stage 1 checkpoint 路径")
    parser.add_argument("--output", required=True, help="输出 checkpoint 路径")
    args = parser.parse_args()

    # 1. 加载 Stage 1 checkpoint
    print(f"[1/4] 加载 Stage 1 checkpoint: {args.source}")
    ckpt = torch.load(args.source, map_location="cpu", weights_only=False)

    # 2. 提取模型权重
    print(f"[2/4] 提取 Actor 权重...")
    old_state = ckpt["model_state_dict"]

    # actor 相关的 key 通常包含 "actor" 字样
    new_state = OrderedDict()
    actor_keys, critic_keys, other_keys = [], [], []
    for k, v in old_state.items():
        if "actor" in k:
            new_state[k] = v.clone()
            actor_keys.append(k)
        elif "critic" in k:
            # Critic 权重丢弃（后续随机初始化）
            critic_keys.append(k)
        else:
            # 其他共享层（如 obs normalization）保留
            new_state[k] = v.clone()
            other_keys.append(k)

    print(f"   ✓ 保留 Actor 层: {len(actor_keys)}")
    print(f"   ✗ 丢弃 Critic 层: {len(critic_keys)} (将随机初始化)")
    print(f"   ✓ 保留其它层:    {len(other_keys)} (如 obs normalization)")

    # 3. 构造新 checkpoint
    print(f"[3/4] 构造新 checkpoint...")
    new_ckpt = {
        "model_state_dict": new_state,
        "epoch": 0,
        "iter": 0,
        "episode_reward_buffer": None,
        "episode_length_buffer": None,
        # 注意: 没有 optimizer_state_dict → 优化器从零开始
    }
    # 保留 obs normalizer 的运行统计量（如有）
    for key in ["actor_obs_rms", "critic_obs_rms"]:
        if key in ckpt:
            new_ckpt[key] = ckpt[key]

    # 4. 保存
    print(f"[4/4] 保存到: {args.output}")
    torch.save(new_ckpt, args.output)
    print("   ✓ 完成！")


if __name__ == "__main__":
    main()
