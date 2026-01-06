from functools import partial
import os
import torch
import torch.nn as nn
import timm.models.vision_transformer
from mmseg.models.builder import BACKBONES


@BACKBONES.register_module()
class MAEVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ ViT backbone for MAE-style features, with optional global avg pooling
        and external pretrained checkpoint loading.
    """
    def __init__(self, global_pool=False, pretrained=None, **kwargs):
        self.pretrained = pretrained
        super(MAEVisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs.get('norm_layer', nn.LayerNorm)
            embed_dim = kwargs.get('embed_dim', getattr(self, 'embed_dim', 768))
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    def init_weights(self, mode='jax'):
        if isinstance(self.pretrained, str) and os.path.isfile(self.pretrained):
            ckpt = torch.load(self.pretrained, map_location='cpu')
            state = ckpt.get('model', ckpt)
            self.load_state_dict(state, strict=False)
        else:
            super().init_weights(mode)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B,1,C]
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]           

        return outcome


def vit_base_patch16(**kwargs):
    model = MAEVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = MAEVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = MAEVisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
