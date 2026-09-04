# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""
自定义动作：手指自由度降维。

分组策略（12D）：拇指4 + 食指+中指共享4 + 无名指+小指共享4。
相邻指共享比原来"中+无+小"跨指共享更合理——Cube 包络抓取时相邻指弯曲角度自然接近。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GroupedHandAction(ActionTerm):
    """手指分组动作：食指+中指共享，无名指+小指共享。"""
    cfg: "GroupedHandActionCfg"

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]

        # 用宽泛模式（thumb.* 等，与 actuator 一致）：兼容新 USD 命名（如 thumb4_hoint 拼写不统一）
        self._thumb_ids, _ = self._asset.find_joints("thumb.*")
        self._index_ids, _ = self._asset.find_joints("index.*")
        self._middle_ids, _ = self._asset.find_joints("middle.*")
        self._ring_ids, _ = self._asset.find_joints("ring.*")
        self._little_ids, _ = self._asset.find_joints("little.*")

        # 打印映射供排查
        print(f"[GroupedHandAction] thumb={list(self._thumb_ids)} index={list(self._index_ids)} "
              f"middle={list(self._middle_ids)} ring={list(self._ring_ids)} little={list(self._little_ids)}")

        self._n_thumb = len(self._thumb_ids)
        self._n_per_finger = 4  # 每指4关节

        # v99: 全部手指关节 id 缓存（scale=0 锁定分支只写手指关节，不碰手臂 6 关节的 position target）
        self._finger_joint_ids = list(self._thumb_ids)
        for _lst in (self._index_ids, self._middle_ids, self._ring_ids, self._little_ids):
            self._finger_joint_ids.extend(_lst)

        self._action_dim = self._n_thumb + 4 + 4   # 拇指 + (食+中) + (无+小)
        self._raw_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)

        # v66: 每步目标变化限幅状态（对齐 SoftHand max_drive_torque_delta 精神，治本防 action_rate 爆炸）
        # 上一步的关节位置目标（num_joints,），限幅基于它而不是 raw action：
        # 即使 std 膨胀导致 raw 大跳，目标每步最多变化 max_delta rad，action_rate 天然有界。
        self._last_target = self._asset.data.joint_pos.clone()
        self._target_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # 食+中 共享映射：action[i] → index_joint_i + middle_joint_i
        self._idx_mid_map = []
        for i in range(self._n_per_finger):
            pair = []
            if i < len(self._index_ids):
                pair.append(self._index_ids[i])
            if i < len(self._middle_ids):
                pair.append(self._middle_ids[i])
            self._idx_mid_map.append(pair)

        # 无+小 共享映射：action[i] → ring_joint_i + little_joint_i
        self._rng_lit_map = []
        for i in range(self._n_per_finger):
            pair = []
            if i < len(self._ring_ids):
                pair.append(self._ring_ids[i])
            if i < len(self._little_ids):
                pair.append(self._little_ids[i])
            self._rng_lit_map.append(pair)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @raw_actions.setter
    def raw_actions(self, value: torch.Tensor):
        self._raw_actions = value

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions + self.cfg.bias
        # v66: raw action clip（对齐 SoftHand clip={.*: (0.0, 1.0)}）——拦截 std 膨胀时 raw 无界大跳。
        # 正常训练 raw 天然在 ±1 附近（RSL-RL 高斯输出），clip 无副作用；
        # 只防探索失控时 raw 到 ±几十 → action_rate 指数放大炸掉。
        if self.cfg.clip_range is not None:
            self._raw_actions = torch.clamp(self._raw_actions, -self.cfg.clip_range, self.cfg.clip_range)

    def apply_actions(self):
        """将策略动作映射为手指关节位置目标（相对位置控制，关节空间 rad）。"""
        actions = self._raw_actions * self.cfg.scale  # 关节空间，单位 rad

        # v99: scale=0（Stage 1 锁定）→ 绝对位置控制：target 固定为初始关节位置，只写手指关节。
        #   原实现 target=joint_pos+0 跟踪当前位置，重力拉下垂后 target 跟着走，
        #   stiffness×(target-pos)=0 → 无恢复力 → 手指自然弯曲（play 观察）。
        #   固定为 default_joint_pos（=init_state.joint_pos，与 reset 后初始姿态一致）后
        #   stiffness 持续产生恢复力对抗重力，保持伸直。
        #   只写手指关节（joint_ids）：手臂 6 关节由 OSC 的 effort target 独立控制，
        #   本 term 只负责手指，不污染手臂的 position target 缓冲区。
        if self.cfg.scale == 0.0:
            dj = self._asset.data.default_joint_pos
            if dj.dim() == 1:
                target = dj[self._finger_joint_ids].unsqueeze(0).expand(
                    self._asset.data.joint_pos.shape[0], -1
                )
            else:
                target = dj[:, self._finger_joint_ids]
            self._asset.set_joint_position_target(target, joint_ids=self._finger_joint_ids)
            self._processed_actions = actions
            return

        thumb_act = actions[:, :self._n_thumb]
        idx_mid_act = actions[:, self._n_thumb:self._n_thumb + 4]
        rng_lit_act = actions[:, -4:]

        full_action = torch.zeros(
            (actions.shape[0], self._asset.num_joints), device=actions.device, dtype=actions.dtype
        )
        # 拇指：翻转 thumb1/2 符号（弯曲方向为负）
        if self._n_thumb > 0:
            thumb_flipped = thumb_act.clone()
            if self._n_thumb >= 2:
                thumb_flipped[:, :2] = -thumb_act[:, :2]
            full_action[:, self._thumb_ids] = thumb_flipped

        # 食+中 共享
        for i, joint_ids in enumerate(self._idx_mid_map):
            if len(joint_ids) > 0:
                full_action[:, joint_ids] = idx_mid_act[:, i:i+1]

        # 无+小 共享
        for i, joint_ids in enumerate(self._rng_lit_map):
            if len(joint_ids) > 0:
                full_action[:, joint_ids] = rng_lit_act[:, i:i+1]

        # [2026-09-02] 绝对位置控制目标 = 伸直位 + 动作：零动作 target=伸直位，stiffness 持续拉回伸直，
        #   消除"相对控制 joint_pos+action 下零动作 target 跟随当前位置 → 手指自由下垂"的问题。
        #   （手臂关节 full_action=0 → target=default_joint_pos，手臂 stiffness=0 无物理作用）
        dj = self._asset.data.default_joint_pos
        if dj.dim() == 1:
            base = dj.unsqueeze(0).expand(actions.shape[0], -1)
        else:
            base = dj
        target = base + full_action

        if self.cfg.max_delta > 0.0:
            # 未初始化的 env 用当前 target 作为起点（首步不限制）
            first = ~self._target_initialized
            if torch.any(first):
                self._last_target[first] = target[first].clone()
                self._target_initialized[first] = True
            # 限幅：target 相对上一步目标最多变化 ±max_delta
            delta = target - self._last_target
            delta = torch.clamp(delta, -self.cfg.max_delta, self.cfg.max_delta)
            target = self._last_target + delta
            self._last_target = target.clone()

        self._asset.set_joint_position_target(target)
        self._processed_actions = actions

    def reset(self, env_ids: slice | torch.Tensor = None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        # v66: 限幅状态按 env 重置（避免跨 episode 的 target 残留拖累新 episode；未 reset 的 env 不受影响）
        self._last_target[env_ids] = self._asset.data.joint_pos[env_ids]
        self._target_initialized[env_ids] = False


@configclass
class GroupedHandActionCfg(ActionTermCfg):
    """手指分组动作的配置类。"""
    class_type: type[ActionTerm] = GroupedHandAction
    asset_name: str = "robot"
    scale: float = 0.05
    bias: float = 0.0  # 手部动作偏置：>0 = 倾向于闭合，降低探索难度
    # [2026-09-02] 0.03→1.5：绝对控制 scale=1.5 后，0.03 远小于单步动作满量程（1.5 rad），
    #   截断高斯动作 → KL/std 失控 → 熵爆（v70 教训：max_delta 必须 ≥ scale 满量程）。
    #   1.5 = scale，不截断正常单步动作，仍拦跨多步极端跳变。Stage 1（scale=0）不走此分支，无影响。
    max_delta: float = 1.0
    # v66: raw action 裁剪范围（对齐 SoftHand clip）。None=不裁剪；1.0=限制 raw∈[-1,1]。
    # 与 max_delta 双保险：clip 拦 raw 绝对值，max_delta 拦每步目标变化。
    clip_range: float | None = 1.0