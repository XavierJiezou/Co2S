import math
from functools import partial
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES
from collections import OrderedDict


# ----------------------------- utils -----------------------------
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or (not training):
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = (qk_scale or head_dim ** -0.5)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x) \
            .reshape(B, N, 3, self.num_heads, C // self.num_heads) \
            .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]       # [B, heads, N, C//heads]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, init_values: float = 0.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)               # [B, C, Hp, Wp]
        Hp, Wp = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # [B, Hp*Wp, C]
        return x, (Hp, Wp)


def _interp_abs_pos_embed(pos_embed: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
    if pos_embed is None:
        return None
    cls_pos, grid = pos_embed[:, :1, :], pos_embed[:, 1:, :]
    N, C = grid.size(1), grid.size(2)
    old_hw = int(round(N ** 0.5))
    if old_hw * old_hw != N and N != Hp * Wp:
        return pos_embed
    if N == Hp * Wp:
        return pos_embed
    grid = grid.reshape(1, old_hw, old_hw, C).permute(0, 3, 1, 2)         # [1, C, h, w]
    grid = F.interpolate(grid, size=(Hp, Wp), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, Hp * Wp, C)
    return torch.cat([cls_pos, grid], dim=1)


def _strip_prefixes(sd, drop_prefixes=('state_dict.', 'module.', 'model.', 'backbone.', 'encoder.')):
    from collections import OrderedDict
    out = OrderedDict()
    for k, v in sd.items():
        nk = k
        for p in drop_prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

def _filter_downstream_heads(sd):
    drop_starts = (
        'head.', 'heads.', 'last_layer.', 'classifier.', 'fc.', 'seg_head.',
        'decode_head.', 'auxiliary_head.', 'logit.', 'proj_head.', 'pred_head.',
        'neck.', 'mlp_head.', 'lm_head.'
    )
    return {k: v for k, v in sd.items() if not k.startswith(drop_starts)}

def _resize_pos_embed_if_needed(src_pe: torch.Tensor, dst_shape: torch.Size) -> torch.Tensor:
    if src_pe is None or src_pe.shape == dst_shape:
        return src_pe
    cls_tok, grid = src_pe[:, :1, :], src_pe[:, 1:, :]
    n_old, c = grid.shape[1], grid.shape[2]
    hw_old = int(round(n_old ** 0.5))
    n_new = dst_shape[1] - 1
    hw_new = int(round(n_new ** 0.5))
    grid = grid.reshape(1, hw_old, hw_old, c).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(grid, size=(hw_new, hw_new), mode='bicubic', align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, n_new, c)
    return torch.cat([cls_tok, grid], dim=1)


# --------------------------- iBOT ViT ----------------------------
@BACKBONES.register_module()
class IBOTVisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 norm_layer=None,
                 init_values: float = 0.0,
                 use_abs_pos_emb: bool = True,
                 out_indices: List[int] = [3, 5, 7, 11],
                 pretrained: str = None,      
                 init_cfg: dict = None,       
                 **kwargs):                  
        super().__init__()
        self.pretrained = pretrained
        self.init_cfg = init_cfg
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
        self.use_abs_pos_emb = use_abs_pos_emb
        self.out_indices = out_indices

        # patch embed
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        # tokens & pos
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values
            ) for i in range(depth)
        ])
        
        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fpn4 = nn.MaxPool2d(kernel_size=4, stride=4)
        else:
            self.fpn1 = nn.Identity()
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.Identity()

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def init_weights(self, pretrained=None):
        path = None
        if isinstance(pretrained, str) and pretrained:
            path = pretrained
        elif isinstance(self.pretrained, str) and self.pretrained:
            path = self.pretrained
        elif isinstance(self.init_cfg, dict):
            ck = self.init_cfg.get('checkpoint', None)
            if isinstance(ck, str) and ck:
                path = ck

        if not path:
            return

        logger = get_root_logger()
        logger.info(f'Loading iBOT backbone ckpt: {path}')

        ckpt = torch.load(path, map_location='cpu')
        for key in ['state_dict', 'student', 'teacher', 'model', 'backbone']:
            if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], (dict,)):
                ckpt = ckpt[key]

        if not isinstance(ckpt, dict):
            from mmcv.runner import load_checkpoint
            load_checkpoint(self, path, strict=False, logger=logger)
            return

        sd = _strip_prefixes(ckpt)
        sd = _filter_downstream_heads(sd)

        model_keys = set(self.state_dict().keys())
        sd = {k: v for k, v in sd.items() if k in model_keys}

        if 'pos_embed' in sd and self.pos_embed is not None:
            try:
                sd['pos_embed'] = _resize_pos_embed_if_needed(sd['pos_embed'], self.pos_embed.shape)
            except Exception as e:
                logger.warning(f'pos_embed resize failed, keep original init. err={e}')

        msg = self.load_state_dict(sd, strict=False)

        ignore_missing_prefix = ('fpn1.', 'fpn2.', 'fpn3.', 'fpn4.',)
        missing_eff = [k for k in msg.missing_keys if not any(k.startswith(p) for p in ignore_missing_prefix)]

        matched = sum(self.state_dict()[k].shape == sd[k].shape for k in sd.keys())
        cov_ckpt = matched / max(1, len(sd))

        if len(missing_eff) == 0:
            logger.info('iBOT ckpt loaded: All backbone keys matched successfully (ignoring FPN*).')
        else:
            logger.info(f'iBOT ckpt loaded: matched={matched} | coverage_ckpt={cov_ckpt:.3f} '
                        f'| missing(effective)={len(missing_eff)}')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, C, H, W = x.shape
        tokens, (Hp, Wp) = self.patch_embed(x)        # [B, Hp*Wp, C]
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B,1,C]
        x = torch.cat([cls_tokens, tokens], dim=1)    # [B, 1+N, C]
        if self.pos_embed is not None:
            pos = _interp_abs_pos_embed(self.pos_embed, Hp, Wp)
            x = x + pos
        x = self.pos_drop(x)

        feats: List[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, self.embed_dim, Hp, Wp)
                feats.append(xp.contiguous())

        # FPN
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(feats)):
            feats[i] = ops[i](feats[i])

        return tuple(feats)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.forward_features(x)
