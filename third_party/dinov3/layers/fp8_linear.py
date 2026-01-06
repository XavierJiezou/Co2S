# third_party/dinov3/layers/fp8_linear.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
兼容版 FP8 Linear

说明:
- 原版 DINOv3 使用 PyTorch 2.x + H100 GPU 的 FP8 加速。
- 在 Python3.8 + PyTorch1.x 环境下, 没有 torch.compiler/torch._dynamo/FP8 支持。
- 本文件提供空实现, 保证 API 不报错，但不会启用 FP8。
"""

import torch
from torch import nn

__all__ = ["convert_linears_to_fp8"]

def convert_linears_to_fp8(model: nn.Module, *args, **kwargs) -> nn.Module:
    """
    空实现: 保持原模型不变，直接返回。
    在不支持 FP8 的环境里，这个函数什么都不做。
    """
    print("[Warning] FP8 is not supported in this environment. Returning model unchanged.")
    return model
