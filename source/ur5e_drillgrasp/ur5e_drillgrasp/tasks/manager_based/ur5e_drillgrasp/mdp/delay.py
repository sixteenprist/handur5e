# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""延迟鲁棒化包装器：观测延迟 + 动作延迟随机化（训练侧，每集/每环境独立采样）。

语义
----
- **观测延迟 D_o**：策略在 t 步拿到的 policy 观测 = 环境 t−D_o 步的原始观测（"信息变旧"）；
  critic 观测**不延迟**（价值估计用当前真实状态，延迟它会拖慢 critic 学习）。
- **动作延迟 D_a**：t 步真正下发给环境的是 t−D_a 步策略算出的动作（"执行变迟"）；
  环境内部 `last_action` 观测量随之自然变成"迟到的命令"——与真机控制器上报一致。
- 每个环境在**每集重置时**独立采样 D_o∈[0, obs_max]、D_a∈[0, act_max]，集内不变
  （文献惯例：OpenAI Dactyl/Rubik 均为每集采样；每步随机会造成非平稳、训不出来）。
- 重置环境：清空缓冲并填入新集初值（观测=新集首帧、动作=0），防跨集泄漏。

用法
----
仅训练用，由环境变量开关（见 ``scripts/rsl_rl/train.py`` 注入点）：
    DELAY_OBS_MAX=2 DELAY_ACT_MAX=1 python scripts/rsl_rl/train.py ... --final
不设置 = 完全不生效（与现状零差异）。评测侧延迟注入在 ``eval_deploy.py``（垫片）。

调试字段（探针用）：``debug_raw_policy`` / ``debug_applied`` / ``debug_flushes``。
"""

import torch
from tensordict import TensorDict  # noqa: F401  （类型标注用）


class DelayVecEnvWrapper:
    """包在 RslRlVecEnvWrapper 外面，拦截 reset/step 注入观测/动作延迟。"""

    def __init__(self, env, obs_max: int = 0, act_max: int = 0):
        self.__dict__["_env"] = env  # __getattr__ 兜底用（先于任何属性访问设置）
        self.obs_max = int(obs_max)  # 0 = 不启用观测延迟
        self.act_max = int(act_max)  # 0 = 不启用动作延迟
        self._L_obs = self.obs_max + 1  # 环长 = 最大延迟+1（要区分 0..max 共 max+1 个年龄）
        self._L_act = self.act_max + 1
        self._num_envs = int(env.num_envs)
        self._device = env.device
        self._t = 0  # 全局步计数（flush 后各环境环的读取语义见 docstring）
        # 缓冲（首次 reset 时按实际维度分配）
        self._obs_dim = None
        self._act_dim = None
        self._obs_ring = None  # (N, obs_max, obs_dim)
        self._act_ring = None  # (N, act_max, act_dim)
        self._d_obs = None  # (N,) long
        self._d_act = None  # (N,) long
        # 调试字段
        self.debug_raw_policy = None
        self.debug_applied = None
        self.debug_flushes = 0

    # ── 属性全部委托给内层 env（num_envs/device/cfg/unwrapped/close…）────────
    def __getattr__(self, name):
        return getattr(self.__dict__["_env"], name)

    # ── 内部工具 ─────────────────────────────────────────────────────────────
    def _alloc(self, obs_dim: int, act_dim: int):
        n = self._num_envs
        self._obs_dim, self._act_dim = obs_dim, act_dim
        if self.obs_max > 0:
            self._obs_ring = torch.zeros(n, self._L_obs, obs_dim, device=self._device)
            self._d_obs = torch.zeros(n, dtype=torch.long, device=self._device)
        if self.act_max > 0:
            self._act_ring = torch.zeros(n, self._L_act, act_dim, device=self._device)
            self._d_act = torch.zeros(n, dtype=torch.long, device=self._device)

    def _sample_delays(self, env_ids: torch.Tensor):
        if self.obs_max > 0:
            self._d_obs[env_ids] = torch.randint(0, self.obs_max + 1, (env_ids.numel(),), device=self._device)
        if self.act_max > 0:
            self._d_act[env_ids] = torch.randint(0, self.act_max + 1, (env_ids.numel(),), device=self._device)

    # ── VecEnv 接口 ──────────────────────────────────────────────────────────
    def reset(self):
        obs_td, extras = self._env.reset()
        raw = obs_td["policy"]
        if self.obs_max > 0 and self._obs_ring is None:
            self._alloc(raw.shape[1], self._act_dim or 22)
        all_ids = torch.arange(self._num_envs, device=self._device)
        self._sample_delays(all_ids)
        if self.obs_max > 0:
            self._obs_ring[:] = raw.unsqueeze(1)  # 全槽填新集首帧
        if self.act_max > 0:
            self._act_ring[:] = 0.0  # 无历史命令
        self._t = 0
        self.debug_raw_policy = raw
        return obs_td, extras

    def step(self, actions: torch.Tensor):
        if self.obs_max > 0 and self._obs_ring is None:  # 防御：未 reset 直接 step
            self._alloc(111, actions.shape[1])
            self._d_obs[:] = 0
        if self.act_max > 0 and self._act_ring is None:
            self._alloc(self._obs_dim or 111, actions.shape[1])
            self._d_act[:] = 0

        self._t += 1
        t = self._t
        n = self._num_envs
        env_idx = torch.arange(n, device=self._device)

        # 1) 动作延迟：先读（上一轮的环），后写（本轮动作）
        if self.act_max > 0:
            idx = (t - self._d_act) % self._L_act
            delayed = self._act_ring[env_idx, idx]
            applied = torch.where((self._d_act > 0).unsqueeze(1), delayed, actions)
        else:
            applied = actions
        self.debug_applied = applied

        # 2) 环境步进
        obs_td, rew, dones, extras = self._env.step(applied)
        raw = obs_td["policy"]
        self.debug_raw_policy = raw

        # 3) 写环
        if self.obs_max > 0:
            self._obs_ring[:, t % self._L_obs] = raw
        if self.act_max > 0:
            self._act_ring[:, t % self._L_act] = actions

        # 4) 重置环境：重采样延迟 + flush 缓冲（避免跨集泄漏）
        done_ids = torch.nonzero(dones > 0).flatten()
        if done_ids.numel() > 0:
            self._sample_delays(done_ids)
            if self.obs_max > 0:
                self._obs_ring[done_ids] = raw[done_ids].unsqueeze(1)
            if self.act_max > 0:
                self._act_ring[done_ids] = 0.0
            self.debug_flushes += int(done_ids.numel())

        # 5) 输出观测：d=0 时读到的正是本轮刚写入的槽，天然正确
        if self.obs_max > 0:
            idx = (t - self._d_obs) % self._L_obs
            obs_out = obs_td.clone()
            obs_out["policy"] = self._obs_ring[env_idx, idx]
        else:
            obs_out = obs_td
        return obs_out, rew, dones, extras

    def close(self):
        return self._env.close()
