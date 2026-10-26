# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""物体形状抽象层（Phase 1：box / sphere）。

目的
----
原观测/奖励里写死了"立方体几何"（半边长 0.03、盒面距离、沿 cube 局部 ±y 的对置轴、
对侧面中心目标）。本模块把这些"形状相关"的量集中为可替换的接口：

- ``configure(kind, size)``：由 env_cfg 在构建时调用（环境变量 OBJECT_KIND / OBJECT_SIZE）；
- ``surface_distance(p, c)``：点到物体表面的距离（box=精确 AABB 距离；sphere=‖p−c‖−r）；
- ``surface_vector(p, c)``：指尖→表面的向量（与旧的球近似同式，box 行为逐位不变）；
- ``surface_offset()``：观测/奖励里用的"表面偏移"（box 半边长 / sphere 半径）。

兼容性
------
- **box 分支保持与改造前完全相同的浮点运算**（逐位一致），由 100 集验收与固定种子
  观测快照对比验证；
- 本阶段一次进程只支持一种形状（通过环境变量设定）；"一个权重通吃多形状"的
  每环境形状张量属于后续 Phase，不在本模块实现。
"""

from __future__ import annotations

import torch

_KIND: str = "box"   # "box" | "sphere"
_SIZE: float = 0.03  # box: 半边长(三轴相同)；sphere: 半径


def configure(kind: str, size: float) -> None:
    """设定当前进程的物体形状（env_cfg 构建时调用一次）。"""
    global _KIND, _SIZE
    kind = str(kind).strip().lower() or "box"
    if kind not in ("box", "sphere"):
        raise ValueError(f"OBJECT_KIND 只支持 box/sphere（当前 {kind!r}）")
    size = float(size)
    if size <= 0.0:
        raise ValueError(f"OBJECT_SIZE 必须 > 0（当前 {size}）")
    _KIND, _SIZE = kind, size


def kind() -> str:
    return _KIND


def size() -> float:
    return _SIZE


def surface_offset() -> float:
    """观测/奖励中的表面偏移：box 半边长 / sphere 半径。"""
    return _SIZE


def is_box() -> bool:
    return _KIND == "box"


def surface_distance(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """(N,3) 点 p 到物体表面（质心 c）的距离 → (N,)。

    box   ：‖clamp(|p−c| − h, min=0)‖（与改造前 _tip_cube_box_dist 完全同式）；
    sphere：clamp(‖p−c‖ − r, min=0)。
    """
    if _KIND == "box":
        dvec = (p - c).abs() - _SIZE
        return torch.norm(dvec.clamp(min=0.0), dim=-1)
    d = torch.norm(p - c, dim=-1)
    return torch.clamp(d - _SIZE, min=0.0)


def surface_vector(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """(N,3) 指尖 p → 表面向量 → (N,3)。

    与旧观测 `fingertip_to_cube_surface` 同式：to_center * clamp((d−offset)/(d+1e-8), min=0)。
    box 时 offset=半边长 → 与改造前逐位一致；sphere 时 offset=半径 → 精确球面。
    """
    to_center = c - p
    dist = torch.norm(to_center, dim=-1, keepdim=True)
    return to_center * torch.clamp((dist - _SIZE) / (dist + 1e-8), min=0.0)
