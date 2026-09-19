from __future__ import annotations

import torch

from isaaclab.managers import CommandTerm


class DrillPoseCommand(CommandTerm):
    """
    目标抓取命令。

    当前（Cube 阶段）：目标固定为桌面上的 Cube 位置。
    后续（电钻阶段）：可以设为电钻开关位置的约束式目标。
    """

    def __init__(self, cfg, env):

        super().__init__(cfg, env)

        # 目标：前3维是位置(x,y,z)，后4维是朝向四元数(w,x,y,z)
        self.goal = torch.zeros(
            (env.num_envs, 7),
            device=env.device
        )

    def reset(self, env_ids=None):
        """兼容旧接口，调用 _resample_command 并返回 metrics."""
        self._resample_command(env_ids)
        return self._update_metrics()

    # ========== 以下是 CommandTerm 要求的三个抽象方法 ==========

    def _resample_command(self, env_ids):
        """重新采样命令（episode 重置时调用）."""
        if env_ids is None:
            env_ids = slice(None)

        # z 0.78 = 桌面 0.75 + 半高 0.03
        # [2026-09-13 方案一] command 冻结：⚠️ 此值仅为"观测兼容"（旧策略熟悉的输入），
        #   不要在这里改"提起高度"！task 目标（离桌 17cm 等）由奖励参数定义
        #   （object_goal_tracking_reward 的 lift_std）。改这里 = 改观测 = 旧策略 OOD → 全局崩。
        self.goal[env_ids, :3] = torch.tensor(
            [-0.16, 0.175, 0.775],
            device=self.goal.device
        )

        # 朝向：单位四元数 (1,0,0,0) = 无旋转
        self.goal[env_ids, 3:] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=self.goal.device
        )

    def _update_command(self):
        """每步更新命令（静态目标无需更新）."""
        pass

    def _update_metrics(self):
        """更新指标（用于调试可视化，目前返回空字典）."""
        return {}

    @property
    def command(self):
        """返回当前目标命令."""
        return self.goal