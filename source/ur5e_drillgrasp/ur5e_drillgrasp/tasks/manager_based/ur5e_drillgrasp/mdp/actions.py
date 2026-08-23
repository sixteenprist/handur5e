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

        self._action_dim = self._n_thumb + 4 + 4   # 拇指 + (食+中) + (无+小)
        self._raw_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)

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

    def apply_actions(self):
        """将策略动作映射为手指关节位置目标（相对位置控制，关节空间 rad）。"""
        actions = self._raw_actions * self.cfg.scale  # 关节空间，单位 rad
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

        self._asset.set_joint_position_target(self._asset.data.joint_pos + full_action)
        self._processed_actions = actions

    def reset(self, env_ids: slice | torch.Tensor = None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class GroupedHandActionCfg(ActionTermCfg):
    """手指分组动作的配置类。"""
    class_type: type[ActionTerm] = GroupedHandAction
    asset_name: str = "robot"
    scale: float = 0.05
    bias: float = 0.0  # 手部动作偏置：>0 = 倾向于闭合，降低探索难度
