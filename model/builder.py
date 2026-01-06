import os
import random
import types
from functools import reduce
import numpy as np

import torch
from mmcv.utils import Config
from mmseg.models import build_segmentor
from mmseg.ops import resize
from torch.nn import functional as F

from model.backbone.timm_vit import TIMMVisionTransformer
from model.decode_heads.dlv3p_head import DLV3PHead
from model.decode_heads.vlg_head import VLGHead

from third_party.dinov3.models.vision_transformer import DinoVisionTransformer
from third_party.maskclip.models.backbones.maskclip_vit import MaskClipVisionTransformer
from third_party.maskclip.models.decode_heads.maskclip_head import MaskClipHead
from third_party.unimatch.model.semseg.deeplabv3plus import DeepLabV3Plus
from third_party.zegclip.losses.atm_loss import SegLossPlus
from third_party.zegclip.models.backbones.clip_vit import CLIPVisionTransformer
from third_party.zegclip.models.backbones.clip_vpt_vit import VPTCLIPVisionTransformer
from third_party.zegclip.models.backbones.text_encoder import CLIPTextEncoder
from third_party.zegclip.models.backbones.utils import DropPath

from model.Co2S import Co2S


def nested_set(dic, key, value):
    keys = key.split('.')
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def nested_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

def is_Co2S(obj):
    return isinstance(obj, Co2S)

def _apply_dropout_to_feats(x, p):
    if torch.is_tensor(x):
        return F.dropout2d(x, p)
    elif isinstance(x, (list, tuple)):
        out = [_apply_dropout_to_feats(e, p) for e in x]
        return type(x)(out) if isinstance(x, tuple) else out
    else:
        return x  

def _concat_fp_feats(x, p):
    if torch.is_tensor(x):
        return torch.cat((x, F.dropout2d(x, p)), dim=0)
    elif isinstance(x, (list, tuple)):
        out = [_concat_fp_feats(e, p) for e in x]
        return type(x)(out) if isinstance(x, tuple) else out
    else:
        return x

def forward_wrapper(self, img, gt=None, need_fp=False, only_fp=False, forward_mode='default'):
    if forward_mode != 'default':
        raise ValueError(forward_mode)

    feats = self.extract_feat(img)

    # disable dropout in modules if requested
    if getattr(self, 'disable_dropout', False):
        dropout_modules = [m for m in self.modules() if isinstance(m, torch.nn.Dropout) or isinstance(m, DropPath)]
        for m in dropout_modules:
            m.eval()

    if is_Co2S(self):
        x = feats  # [visual, text, conv]
        if only_fp or need_fp:
            pyramid = x[0][0]                 # list of 4D maps
            text_feats = x[1]                 # [B,N,C] or [N,C]
            conv_feats = x[2] if len(x) > 2 else None

        if only_fp:
            x[0][0] = _apply_dropout_to_feats(pyramid, getattr(self, 'fp_rate', 0.5))
            if conv_feats is not None:
                x[2] = _apply_dropout_to_feats(conv_feats, getattr(self, 'fp_rate', 0.5))
            if hasattr(self.decode_head, 'dc_unperturbed') and self.decode_head.dc_unperturbed:
                assert len(x[0]) == 2
                x[0].append(pyramid)

            out = self._decode_head_forward_test(x, img_metas=None)
            out = resize(input=out, size=img.shape[2:], mode='bilinear',
                         align_corners=getattr(self, 'align_corners', False))
            return out

        if need_fp:
            x[0][0] = _concat_fp_feats(pyramid, getattr(self, 'fp_rate', 0.5))
            if isinstance(text_feats, torch.Tensor):
                if text_feats.dim() == 3:
                    x[1] = torch.cat((text_feats, text_feats), dim=0)
                elif text_feats.dim() == 2:
                    x[1] = text_feats
                else:
                    raise RuntimeError(f"Unexpected text_feats.dim={text_feats.dim()}")
            if conv_feats is not None:
                x[2] = _concat_fp_feats(conv_feats, getattr(self, 'fp_rate', 0.5))
            if hasattr(self.decode_head, 'dc_unperturbed') and self.decode_head.dc_unperturbed:
                assert len(x[0]) == 2
                x[0].append([torch.cat((f, f), dim=0) for f in pyramid])

            out = self._decode_head_forward_test(x, img_metas=None)
            out = resize(input=out, size=img.shape[2:], mode='bilinear',
                         align_corners=getattr(self, 'align_corners', False))
            v_clean, v_fp = out.chunk(2, dim=0)
            return v_clean, v_fp

        out = self._decode_head_forward_test(x, img_metas=None)
        out = resize(input=out, size=img.shape[2:], mode='bilinear',
                     align_corners=getattr(self, 'align_corners', False))
        return out

    x = feats
    if only_fp:
        x = [F.dropout2d(f, getattr(self, 'fp_rate', 0.5)) for f in x]
    elif need_fp:
        x = [torch.cat((f, F.dropout2d(f, getattr(self, 'fp_rate', 0.5))), dim=0) for f in x]

    out = self._decode_head_forward_test(x, img_metas=None)
    out = resize(input=out, size=img.shape[2:], mode='bilinear',
                 align_corners=getattr(self, 'align_corners', False))
    if need_fp:
        out = out.chunk(2, dim=0)
    return out

def _default_text_emb_path(dataset: str) -> str:
    ds = dataset.lower()
    base = 'configs/_base_/datasets/text_embedding'
    if ds == 'whdld':
        return f'{base}/whdld_conceptavg6_single.npy'
    elif ds == 'loveda':
        return f'{base}/loveda_conceptavg5_single.npy'
    elif ds == 'potsdam':
        return f'{base}/potsdam_conceptavg5_single.npy'
    elif ds == 'gid':
        return f'{base}/gid_conceptavg6_single.npy'
    elif ds == 'mer' or 'msl':
        return f'{base}/mer_conceptavg6_single.npy'
    else:
        raise ValueError(f'Unknown dataset for default text embedding: {dataset}')

def build_model(cfg):
    model_type = cfg['model']

    if model_type == 'deeplabv3plus':
        model = DeepLabV3Plus(cfg)
        if 'disable_dropout' in cfg:
            model.disable_dropout = cfg['disable_dropout']
        if 'fp_rate' in cfg:
            model.fp_rate = cfg['fp_rate']
        return model

    if 'mmseg.' in model_type:
        model_type = model_type.replace('mmseg.', '')
        model_cfg_file = f'configs/_base_/models/{model_type}.py'
        mmseg_cfg = Config.fromfile(model_cfg_file)

        # nclass by dataset
        if cfg['dataset'] == 'whdld':
            cfg['nclass'] = 6
        elif cfg['dataset'] == 'loveda':
            cfg['nclass'] = 7
        elif cfg['dataset'] == 'potsdam':
            cfg['nclass'] = 6
        elif cfg['dataset'] == 'gid':
            cfg['nclass'] = 15
        elif cfg['dataset'] == 'mer' or 'msl':
            cfg['nclass'] = 9
        else:
            raise ValueError(f'Unknown dataset: {cfg["dataset"]}')
        if 'decode_head' in mmseg_cfg['model']:
            mmseg_cfg['model']['decode_head']['num_classes'] = cfg['nclass']

        # align image size and inner fields
        if mmseg_cfg.get('img_size', None) != cfg['crop_size']:
            nested_set(mmseg_cfg, 'img_size', cfg['crop_size'])
            if nested_get(mmseg_cfg, 'model.backbone', None) is not None:
                nested_set(mmseg_cfg, 'model.backbone.img_size', (cfg['crop_size'], cfg['crop_size']))
            if nested_get(mmseg_cfg, 'model.dino_backbone', None) is not None and \
               'img_size' in mmseg_cfg['model']['dino_backbone']:
                nested_set(mmseg_cfg, 'model.dino_backbone.img_size', (cfg['crop_size'], cfg['crop_size']))
            if nested_get(mmseg_cfg, 'model.decode_head', None) is not None:
                nested_set(mmseg_cfg, 'model.decode_head.img_size', cfg['crop_size'])

        # pass-through extra model args
        if 'model_args' in cfg:
            mmseg_cfg['model'].update(cfg['model_args'])

        if mmseg_cfg['model'].get('type') == 'Co2S':
            backbone_type = mmseg_cfg['model'].get('backbone_type', 'clip')
            if backbone_type == 'clip':
                emb_path = cfg.get('text_embedding_path', None) or _default_text_emb_path(cfg['dataset'])
                if mmseg_cfg['model'].get('load_text_embedding', None) is None and \
                   mmseg_cfg['model'].get('text_embedding', None) is None:
                    nested_set(mmseg_cfg, 'model.load_text_embedding', emb_path)
                mcc_path = cfg.get('mcc_text_embedding_path', None) or emb_path
                pl_path  = cfg.get('pl_text_embedding_path',  None) or emb_path
                if mmseg_cfg['model'].get('load_mcc_text_embedding', None) is None:
                    nested_set(mmseg_cfg, 'model.load_mcc_text_embedding', mcc_path)
                if mmseg_cfg['model'].get('load_pl_text_embedding', None) is None:
                    nested_set(mmseg_cfg, 'model.load_pl_text_embedding',  pl_path)

        # build
        model = build_segmentor(
            mmseg_cfg.model,
            train_cfg=mmseg_cfg.get('train_cfg'),
            test_cfg=mmseg_cfg.get('test_cfg'))

        # common flags
        model.disable_dropout = cfg['disable_dropout']
        model.fp_rate = cfg['fp_rate']

        # inject our forward wrapper
        model.forward = types.MethodType(forward_wrapper, model)
        model.init_weights()
        return model

    raise ValueError(f'Unsupported model type: {model_type}')
