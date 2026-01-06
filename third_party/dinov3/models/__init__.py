# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from third_party.dinov3.layers.fp8_linear import convert_linears_to_fp8
from . import vision_transformer as vits

logger = logging.getLogger("dinov3")


def init_fp8(model: nn.Module, args) -> nn.Module:
    if not getattr(args, "fp8_enabled", False):
        logger.info("fp8 matmuls: OFF (disabled in config)")
        return model

    if not hasattr(torch, "_inductor"):
        logger.warning("FP8 not supported in this PyTorch version. Skipping FP8 init.")
        return model

    logger.info("fp8 matmuls: ON")
    try:
        torch._inductor.config.triton.multi_kernel = 1
    except Exception:
        logger.warning("torch._inductor.config not available, skipping multi_kernel setting.")

    return convert_linears_to_fp8(model, filter=getattr(args, "fp8_filter", ".*"))


def build_model(args, only_teacher=False, img_size=224, device=None):
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            pos_embed_rope_base=args.pos_embed_rope_base,
            pos_embed_rope_min_period=args.pos_embed_rope_min_period,
            pos_embed_rope_max_period=args.pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=args.pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=args.pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=args.pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=args.pos_embed_rope_rescale_coords,
            qkv_bias=args.qkv_bias,
            layerscale_init=args.layerscale,
            norm_layer=args.norm_layer,
            ffn_layer=args.ffn_layer,
            ffn_bias=args.ffn_bias,
            proj_bias=args.proj_bias,
            n_storage_tokens=args.n_storage_tokens,
            mask_k_bias=args.mask_k_bias,
            untie_cls_and_patch_norms=args.untie_cls_and_patch_norms,
            untie_global_and_local_cls_norm=args.untie_global_and_local_cls_norm,
            device=device,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        teacher = init_fp8(teacher, args)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
        )
        embed_dim = student.embed_dim
    else:
        raise NotImplementedError(f"Unrecognized architecture {args.arch}")
    student = init_fp8(student, args)
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher: bool = False):
    outputs = build_model(
        cfg.student,
        only_teacher=only_teacher,
        img_size=cfg.crops.global_crops_size
        if isinstance(cfg.crops.global_crops_size, int)
        else max(cfg.crops.global_crops_size),
        device="meta",
    )
    if only_teacher:
        teacher, embed_dim = outputs
        return teacher, embed_dim
    else:
        student, teacher, embed_dim = outputs
        return student, teacher, embed_dim


def build_model_for_eval(
    config,
    pretrained_weights: Union[str, Path] = None,
    shard_unsharded_model: bool = False,
):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    if not pretrained_weights:
        logger.info("No pretrained weights")
        model.init_weights()
    elif Path(pretrained_weights).is_dir():
        logger.info("PyTorch DCP checkpoint")

        try:
            from third_party.dinov3.checkpointer import load_checkpoint
            from third_party.dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
        except ImportError:
            logger.warning("Checkpointer/FSDP not available in this environment. Skipping checkpoint load.")
            return model

        moduledict = nn.ModuleDict({"backbone": model})
        ac_compile_parallelize(moduledict, inference_only_models=[], cfg=config)
        model.to_empty(device="cuda")
        load_checkpoint(pretrained_weights, model=moduledict, strict_loading=True)
        shard_unsharded_model = False
    else:
        logger.info("PyTorch consolidated checkpoint")
        try:
            from third_party.dinov3.checkpointer import init_model_from_checkpoint_for_evals
        except ImportError:
            logger.warning("Checkpointer not available in this environment. Skipping checkpoint load.")
            return model

        model.to_empty(device="cuda")
        init_model_from_checkpoint_for_evals(model, pretrained_weights, "teacher")

    if shard_unsharded_model:
        logger.info("Sharding model")
        try:
            from third_party.dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
            moduledict = nn.ModuleDict({"backbone": model})
            ac_compile_parallelize(moduledict, inference_only_models=[], cfg=config)
        except ImportError:
            logger.warning("FSDP not available. Skipping sharding.")

    model.eval()
    return model
