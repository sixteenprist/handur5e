# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""
自定义动作：手指自由度降维。

分组策略（8D）：拇指4（独立） + 中指4（独立）。
[2026-09-22 v7.10 用户要求] 由"四指共享4"改为"仅中指4"：任务约定只有拇指+中指参与
  （其余三指资产里无碰撞），共享通道会让食/无/小跟随摆动纯属视觉噪音 → 现在
  index/ring/little 的目标恒为初始位姿（full_action=0），只有中指逐节卷曲。
  动作维度仍是 8（拇指4+中指4）、观测维度不变 → 旧 ckpt 兼容（行为等价，三指静止）。
层级弯曲是"跨关节"：中指 4 关节仍有 4 个独立通道，四通道值可不同 → 逐节卷曲包络
（v18 教训：掌根弯、中段不弯 → 指尖够不到 → 挤飞；切勿把 4 通道再压成 1 个）。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GroupedHandAction(ActionTerm):
    """手指分组动作：拇指独立 4 通道，四指（食/中/无/小）跨指共享 4 通道。"""
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

        self._action_dim = self._n_thumb + 4   # 拇指4(独立) + 四指共享4
        self._raw_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)

        # v66: 每步目标变化限幅状态（对齐 SoftHand max_drive_torque_delta 精神，治本防 action_rate 爆炸）
        # 上一步的关节位置目标（num_joints,），限幅基于它而不是 raw action：
        # 即使 std 膨胀导致 raw 大跳，目标每步最多变化 max_delta rad，action_rate 天然有界。
        self._last_target = self._asset.data.joint_pos.clone()
        self._target_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # [v7.10] 仅中指接受四指通道；index/ring/little 不再映射（目标恒为初值，保持静止）

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
        """将策略动作映射为手指关节位置目标（绝对位置控制：伸直位 + 动作，关节空间 rad）。"""
        actions = self._raw_actions * self.cfg.scale  # 关节空间，单位 rad

        thumb_act = actions[:, :self._n_thumb]
        four_act = actions[:, self._n_thumb:self._n_thumb + 4]

        full_action = torch.zeros(
            (actions.shape[0], self._asset.num_joints), device=actions.device, dtype=actions.dtype
        )
        # 拇指：翻转 thumb1/2 符号（弯曲方向为负）
        if self._n_thumb > 0:
            thumb_flipped = thumb_act.clone()
            if self._n_thumb >= 2:
                thumb_flipped[:, :2] = -thumb_act[:, :2]
            full_action[:, self._thumb_ids] = thumb_flipped

        # [v7.10 用户要求] 四指通道只写"中指"；其余三指 full_action 保持 0 → target=default（初始位姿）
        if len(self._middle_ids) > 0:
            n_m = min(4, len(self._middle_ids))
            full_action[:, self._middle_ids[:n_m]] = four_act[:, :n_m]

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
    # [2026-09-15 对齐·防呆] 原默认值 0.05，但 env_cfg 一直显式传入 1.0（实际生效值 = 1.0；
    #   0.05 从未生效，曾被误读为"手指动程仅 ±3°"的警报源）。改为与实例值一致，防再次误读。
    scale: float = 1.0
    bias: float = 0.0  # 手部动作偏置（降低探索难度）；⚠️ 各通道弯曲符号不一（拇指1/2 已翻转），
    #   启用前先验证"正值=闭合"的方向
    # 每步目标位置最大变化量 (rad/step)，防"跨步极端跳变"并柔化接触（慢速压进）。
    # [2026-09-14 修正] 1.0→0.5：原值与 scale 满量程相等（±1.0 rad），而步间跳变最大 2.0 rad，
    #   正常动作变化（~0.3-0.7）完全不受约束 → 限幅名存实亡（用户发现）。
    #   0.5 = 只截"半步以上"的极端跳变，并天然柔化接触压进速度。
    #   调节方向：S2 接触冲击大（弹开 cube）→ 收紧 0.3；手部响应迟钝/截断过频 → 放宽 0.7。
    #   ⚠️ 不要 <0.2：限幅过紧会截断探索动作 → 策略行为与输出脱节、std 膨胀（v70 教训的另一面）。
    max_delta: float = 0.5
    # v66: raw action 裁剪范围（对齐 SoftHand clip）。None=不裁剪；1.0=限制 raw∈[-1,1]。
    # 与 max_delta 双保险：clip 拦 raw 绝对值，max_delta 拦每步目标变化。
    # [2026-09-19 力封闭配方] 1.0→1.6：硬编程实测（scripted_grasp v0.18-v0.21）——手指关节
    #   限位 ±1.57（j2/j3/j4 可达 1.4+），而 raw clip=1.0 把目标卡在 default±1.0 rad（57°），
    #   四指/拇指"深弯到贴面/绕位"所需行程用不满 → 接触力上不去（历史 opp 仅 0.06~0.28N）。
    #   1.6 覆盖全部行程（thumb1 限位 -1.57、t2/t3/t4 均 ±1.57）；与 scripted_grasp 的
    #   --hand_clip 默认值 1.6 对齐。回退：改回 1.0（策略行为会变，需重训）。
    clip_range: float | None = 1.6


# ══════════════════════════════════════════════════════════════
# [2026-09-17 防飞双保险] 手臂 OSC 安全子类：raw clip + 步间变化限幅
# ══════════════════════════════════════════════════════════════
# 背景：Isaac Lab 原生 OperationalSpaceControllerAction 没有任何出口限幅——
#   _preprocess_actions 只做 ×scale（position/orientation_scale=0.005）；
#   cfg.clip 在该类里不生效（仅 IK 动作类会 clamp，见 task_space_actions.py 对应实现）。
#   于是 σ 爆/值崩时 raw 可达 ±5~10 → 每步目标乱拽（±2.5~5cm、±0.025~0.05rad/步）→
#   手臂乱飞乱晃。手部早有 v66 双保险（clip ±1.0 + max_delta 0.5），手臂一直裸奔。
# 本子类补上同款双保险（对齐 SoftHand 的 max_drive_torque_delta / clip_actions 精神）：
#   ① raw_clip：拦 raw 绝对值（σ 膨胀时截断幅度）；
#   ② max_delta：拦步间变化（治"每步方向乱反"的晃，兼作目标加速度限幅）。
#   顺序：先幅度、后步间；_last_raw 记录"实际生效"的 raw（防 windup）。
# 正常训练零影响：阈值只截极端（正常臂 raw ~±0.5~1.5、步间变化 ~≤0.5~1.0）；
#   上线后按"正常阶段从不触发"核对。调参方向：还晃→收紧；动作迟钝→放宽。
# checkpoint 兼容：无参数形状变化，直接 resume。


class SafeOperationalSpaceControllerAction(OperationalSpaceControllerAction):
    """OSC 动作 + raw 幅度/步间变化双限幅（防崩坏时手臂乱飞乱晃）。"""

    cfg: "SafeOperationalSpaceControllerActionCfg"

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # 上一步"实际生效"的 raw（限幅状态；跨 episode 由 reset 清零）
        self._last_raw = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    def _preprocess_actions(self, actions: torch.Tensor):
        a = actions
        # ① raw 幅度限幅（σ 膨胀时拦无界大跳）
        if self.cfg.raw_clip is not None and self.cfg.raw_clip > 0.0:
            a = torch.clamp(a, -self.cfg.raw_clip, self.cfg.raw_clip)
        # ② 步间变化限幅（治高频反向乱晃）
        if self.cfg.max_delta is not None and self.cfg.max_delta > 0.0:
            delta = torch.clamp(a - self._last_raw, -self.cfg.max_delta, self.cfg.max_delta)
            a = self._last_raw + delta
        self._last_raw.copy_(a)
        # 交回父类做 ×scale 与 OSC 命令设置
        super()._preprocess_actions(a)

    def reset(self, env_ids: slice | torch.Tensor = None):
        super().reset(env_ids)
        self._last_raw[env_ids] = 0.0


@configclass
class SafeOperationalSpaceControllerActionCfg(OperationalSpaceControllerActionCfg):
    """SafeOperationalSpaceControllerAction 的配置（继承原生 OSC cfg，新增两个限幅参数）。"""

    class_type: type[ActionTerm] = SafeOperationalSpaceControllerAction
    # raw action 绝对值上限（None/<=0 = 关闭）。
    #   建议 2.0 起步（正常臂 raw 基本 ≤1.5；σ 爆时 raw ±5~10 会被截到 ±2）。
    raw_clip: float | None = 2.5
    # 每步 raw 变化上限（None/<=0 = 关闭）。
    #   建议 1.5 起步（正常步间变化 ≤~1.0；崩坏乱跳时截到 1.5/步）。
    max_delta: float | None = 2.0
