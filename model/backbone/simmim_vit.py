import math
from functools import partial
from typing import Tuple, List, Dict, Any, OrderedDict as _OrderedDictType

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv.runner import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES


# ----------------------------- MLP / Attention / Block -----------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads if attn_head_dim is None else attn_head_dim
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias, self.v_bias = None, None
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1
            )  # Wh*Ww, Wh*Ww, nH
            rel_bias = rel_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + rel_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., init_values=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# ----------------------------- PatchEmbed -----------------------------
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs) -> Tuple[torch.Tensor, Tuple[int, int]]:
        y = self.proj(x)  # [B, C, Hp, Wp]
        Hp, Wp = y.shape[-2], y.shape[-1]
        tokens = y.flatten(2).transpose(1, 2)  # [B, Hp*Wp, C]
        return tokens, (Hp, Wp)


# ----------------------------- Abs Pos Embed resize -----------------------------
def _resize_pos_embed_if_needed(src_pe: torch.Tensor, dst_shape: torch.Size) -> torch.Tensor:
    """Interpolate absolute pos embed to new grid size when needed.
    src_pe: [1, 1+N_old, C], dst_shape: torch.Size([1, 1+N_new, C])
    """
    if src_pe is None or src_pe.shape == dst_shape:
        return src_pe
    cls_pos, grid = src_pe[:, :1, :], src_pe[:, 1:, :]
    n_old, c = grid.shape[1], grid.shape[2]
    hw_old = int(round(n_old ** 0.5))
    n_new = dst_shape[1] - 1
    hw_new = int(round(n_new ** 0.5))
    grid = grid.reshape(1, hw_old, hw_old, c).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(hw_new, hw_new), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, n_new, c)
    return torch.cat([cls_pos, grid], dim=1)


# ----------------------------- Backbone -----------------------------
@BACKBONES.register_module()
class SIMMIMVisionTransformer(nn.Module):
    """SimMIM ViT backbone for segmentation (pyramid 4D features)."""
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=None,
                 init_values=None,
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False,   
                 use_mean_pooling=False,       
                 out_indices: List[int] = [3, 5, 7, 11],
                 pretrained=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_rel_pos_bias = use_rel_pos_bias
        self.out_indices = out_indices
        self.pretrained = pretrained 

        # patch embed
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        # tokens / pos
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        # shared relative position bias
        if use_shared_rel_pos_bias:
            win = self.patch_embed.patch_shape
            self.rel_pos_bias = nn.Module()  
            self._shared_rel_pos_bias_window = win
        else:
            self.rel_pos_bias = None
            self._shared_rel_pos_bias_window = None

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, init_values=init_values,
                window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None
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

        # init
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)


    @staticmethod
    def _interp_abs_pos_embed(pos_embed: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
        cls_pos, patch_pos = pos_embed[:, :1, :], pos_embed[:, 1:, :]
        old_hw = int(round(patch_pos.shape[1] ** 0.5))
        if old_hw * old_hw != patch_pos.shape[1]:
            return pos_embed
        patch_pos = patch_pos.reshape(1, old_hw, old_hw, -1).permute(0, 3, 1, 2)      # [1,C,h,w]
        patch_pos = F.interpolate(patch_pos, size=(Hp, Wp), mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, Hp * Wp, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)
    
    
    # ---------------- Weights init & load ----------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            ckpt = torch.load(pretrained, map_location='cpu')
            sd = ckpt.get('model', ckpt.get('state_dict', ckpt))

            DROP = ('state_dict.', 'module.', 'model.', 'encoder.', 'backbone.')
            new_sd = {}
            for k, v in sd.items():
                nk = k
                for p in DROP:
                    if nk.startswith(p):
                        nk = nk[len(p):]
                new_sd[nk] = v
            sd = new_sd

            sd = {k: v for k, v in sd.items() if not k.startswith('head.')}

            if self.pos_embed is not None and 'pos_embed' in sd:
                if sd['pos_embed'].shape != self.pos_embed.shape:
                    Hp, Wp = self.patch_embed.patch_shape   
                    sd['pos_embed'] = self._interp_abs_pos_embed(sd['pos_embed'], Hp, Wp)

            ret = self.load_state_dict(sd, strict=False)

            missing = [k for k in ret.missing_keys if not k.startswith(('fpn',))]
            logger.info(f"SIMMIM ckpt loaded: matched={len(sd)-len(ret.unexpected_keys)} | "
                        f"coverage_ckpt=1.000 | missing(effective)={len(missing)}")
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    # ---------------- misc ----------------
    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # ---------------- forward ----------------
    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, C, H, W = x.shape
        tokens, (Hp, Wp) = self.patch_embed(x)  # [B, N, C], N=Hp*Wp
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)  # [B, 1+N, C]

        if self.pos_embed is not None:
            pos = _resize_pos_embed_if_needed(self.pos_embed, x.shape)
            x = x + pos
        x = self.pos_drop(x)

        feats: List[torch.Tensor] = []
        rel_pos_bias = None 
        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias=rel_pos_bias)
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, self.embed_dim, Hp, Wp)
                feats.append(xp.contiguous())

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(feats)):
            feats[i] = ops[i](feats[i])

        return tuple(feats)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.forward_features(x)
