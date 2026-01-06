import logging
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from typing_extensions import Literal

import torch
import torch.nn.init
from torch import Tensor, nn

from third_party.dinov3.layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from third_party.dinov3.utils import named_apply

from mmseg.models.builder import BACKBONES


from model.backbone.lora import LoRA

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


# @BACKBONES.register_module()
class DinoVisionTransformer(nn.Module):
    """
    DINOv3 ViT 主干（改造版）：
      1) 支持 LoRA（q/k/v/o 任意子集），通过 forward hook 以增量注入；
      2) 支持按 block 索引冻结（只冻基座参数，LoRA 仍可训）；
      3) 兼容 qkv 合并/拆分两种实现；
      4) 不破坏原有前向/权重加载。

    额外新增 init 参数（保持向后兼容）：
      - lora_layers: Optional[List[int]]  需要注入 LoRA 的 block 索引（从 0 开始）
      - lora_r: int = 4
      - lora_scaling: float = 1.0
      - lora_dropout: float = 0.0
      - lora_targets: str = "qkvo"       # 子集如 "qv"、"qo" 等
      - freeze_block_indices: Optional[List[int]] = None
      - freeze_patch_embed: bool = False
      - freeze_cls_token: bool = False
      - norm_eval_in_frozen: bool = True
    """
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: Optional[float] = None,
        pos_embed_rope_max_period: Optional[float] = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: Optional[float] = None,
        pos_embed_rope_jitter_coords: Optional[float] = None,
        pos_embed_rope_rescale_coords: Optional[float] = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: Optional[float] = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Optional[Any] = None,

        # ===== 新增：LoRA 配置 =====
        lora_layers: Optional[List[int]] = None,
        lora_r: int = 4,
        lora_scaling: float = 1.0,
        lora_dropout: float = 0.0,
        lora_targets: str = "qkvo",

        # ===== 新增：冻结配置 =====
        freeze_block_indices: Optional[List[int]] = None,
        freeze_patch_embed: bool = False,
        freeze_cls_token: bool = False,
        norm_eval_in_frozen: bool = True,

        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        # ===== 保存 LoRA/冻结配置 =====
        self._lora_layers = set(lora_layers or [])
        self._lora_cfg = dict(
            r=lora_r, scaling=lora_scaling, dropout=lora_dropout, targets=lora_targets
        )
        self._lora_handles: List[Any] = []
        self._freeze_block_indices = set(freeze_block_indices or [])
        self._freeze_patch_embed = bool(freeze_patch_embed)
        self._freeze_cls_token = bool(freeze_cls_token)
        self._norm_eval_in_frozen = bool(norm_eval_in_frozen)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        if len(self._lora_layers) > 0:
            for idx, blk in enumerate(self.blocks):
                if idx in self._lora_layers:
                    self._inject_lora_into_block(blk, embed_dim)

        if (
            len(self._freeze_block_indices) > 0
            or self._freeze_patch_embed
            or self._freeze_cls_token
        ):
            self._apply_freeze_settings()


    def _inject_lora_into_block(self, blk: nn.Module, dim: int) -> None:
        """对一个 SelfAttentionBlock 注入 LoRA。
        兼容两种注意力实现：
          1) 合并 qkv 的 Linear，形状 [3*dim, dim]
          2) 拆分 q_proj/k_proj/v_proj 的三个 Linear，形状各 [dim, dim]
        同时对输出 proj（[dim, dim]）加 'o' LoRA（若启用）。
        """
        # 生成 LoRA 适配器，并迁移到 block 的 dtype/device
        lora = LoRA(
            dim=dim,
            r=self._lora_cfg["r"],
            scaling=self._lora_cfg["scaling"],
            dropout=self._lora_cfg["dropout"],
            targets=self._lora_cfg["targets"],
        )
        # 迁移 dtype/device
        ref_param = next(blk.parameters())
        lora.to(device=ref_param.device, dtype=ref_param.dtype)
        # 将 LoRA 挂在 block 上，保证参数被注册
        blk._lora_adapter = lora

        # ---- hook 定义 ----
        def _qkv_hook(mod: nn.Linear, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            # inputs[0]: x, shape [B, N, dim]
            x = inputs[0]
            B, N, C = x.shape
            # 逐分量构造增量
            dq = x.new_zeros(B, N, C)
            dk = x.new_zeros(B, N, C)
            dv = x.new_zeros(B, N, C)
            if 'q' in lora.targets:
                dq = lora.b_q(lora.a_q(lora.dropout(x))) * lora.scaling
            if 'k' in lora.targets:
                dk = lora.b_k(lora.a_k(lora.dropout(x))) * lora.scaling
            if 'v' in lora.targets:
                dv = lora.b_v(lora.a_v(lora.dropout(x))) * lora.scaling
            delta_qkv = torch.cat([dq, dk, dv], dim=-1)  # [B, N, 3*dim]
            return output + delta_qkv

        def _q_hook(mod: nn.Linear, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if 'q' not in lora.targets:
                return output
            x = inputs[0]
            return output + (lora.b_q(lora.a_q(lora.dropout(x))) * lora.scaling)

        def _k_hook(mod: nn.Linear, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if 'k' not in lora.targets:
                return output
            x = inputs[0]
            return output + (lora.b_k(lora.a_k(lora.dropout(x))) * lora.scaling)

        def _v_hook(mod: nn.Linear, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if 'v' not in lora.targets:
                return output
            x = inputs[0]
            return output + (lora.b_v(lora.a_v(lora.dropout(x))) * lora.scaling)

        def _proj_hook(mod: nn.Linear, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if 'o' not in lora.targets:
                return output
            y = inputs[0]  # attn 输出输入到 proj 的张量 [B, N, dim]
            return output + (lora.b_o(lora.a_o(lora.dropout(y))) * lora.scaling)

        # ---- 自动定位 qkv/proj 并注册 hook ----
        qkv_found = False
        q_found = k_found = v_found = False
        proj_found = False

        for name, m in blk.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            in_f, out_f = m.in_features, m.out_features

            # 优先匹配 “合并 qkv”
            if (not qkv_found) and in_f == dim and out_f == 3 * dim and ('qkv' in name or 'in_proj' in name):
                self._lora_handles.append(m.register_forward_hook(_qkv_hook))
                qkv_found = True
                continue

            # 匹配拆分 q/k/v
            lname = name.lower()
            if (not q_found) and in_f == dim and out_f == dim and any(t in lname for t in ['q_proj', '.q', 'to_q']):
                self._lora_handles.append(m.register_forward_hook(_q_hook))
                q_found = True
                continue
            if (not k_found) and in_f == dim and out_f == dim and any(t in lname for t in ['k_proj', '.k', 'to_k']):
                self._lora_handles.append(m.register_forward_hook(_k_hook))
                k_found = True
                continue
            if (not v_found) and in_f == dim and out_f == dim and any(t in lname for t in ['v_proj', '.v', 'to_v']):
                self._lora_handles.append(m.register_forward_hook(_v_hook))
                v_found = True
                continue

        # proj
        for name, m in blk.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            in_f, out_f = m.in_features, m.out_features
            lname = name.lower()
            if in_f == dim and out_f == dim and any(t in lname for t in ['proj', 'out_proj', 'to_out']):
                self._lora_handles.append(m.register_forward_hook(_proj_hook))
                proj_found = True
                break

        if not (qkv_found or (q_found and k_found and v_found)):
            logger.warning("LoRA injection: qkv/q,k,v Linear not found in a block; skip attention LoRA for this block.")
        if ('o' in lora.targets) and (not proj_found):
            logger.warning("LoRA injection: proj Linear not found in a block; skip o-LoRA for this block.")

    # ===================== 冻结：与 CLIP 同步冻结相同层 =====================

    def _apply_freeze_settings(self) -> None:
        """按配置冻结参数：只冻结基座权重，保留 LoRA 可训练。"""
        # 冻结 patch_embed / cls_token
        if self._freeze_patch_embed:
            for p in self.patch_embed.parameters():
                p.requires_grad = False
        if self._freeze_cls_token:
            self.cls_token.requires_grad = False
            if hasattr(self, "storage_tokens"):
                self.storage_tokens.requires_grad = False

        # 冻结指定 blocks（除了 LoRA 参数）
        for idx, blk in enumerate(self.blocks):
            if idx in self._freeze_block_indices:
                for n, p in blk.named_parameters():
                    # LoRA 参数放行
                    if "_lora_adapter" in n:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

    # ===================== 初始化 & 前向 =====================

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Union[Tensor, List[Tensor]], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, is_training: bool = False, **kwargs) -> Union[List[Dict[str, Tensor]], Tensor]:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

    # ===================== 训练态行为（Norm eval 于冻结块） =====================

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            return self
        if self._norm_eval_in_frozen and (len(self._freeze_block_indices) > 0):
            for idx, blk in enumerate(self.blocks):
                if idx in self._freeze_block_indices:
                    for m in blk.modules():
                        # 冻结块内的归一化直接 eval，避免跑动量/统计
                        if isinstance(m, (nn.LayerNorm, RMSNorm)):
                            m.eval()
        return self


def vit_small(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model
