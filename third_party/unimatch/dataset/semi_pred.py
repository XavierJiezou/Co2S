from .transform import *

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from mmseg.datasets.pipelines.transforms import Resize


class SemiDataset(Dataset):
    def __init__(self, cfg, mode='pred', id_path=None):
        assert mode in ('pred', 'val'), "SemiPredDataset only support pred / val"
        self.name = cfg['dataset']
        self.root = os.path.expandvars(os.path.expanduser(cfg['data_root']))
        self.mode = mode
        self.size = cfg['crop_size']              
        self.img_scale = cfg.get('img_scale', None)
        self.scale_ratio_range = cfg.get('scale_ratio_range', (0.5, 2.0))
        self.reduce_zero_label = cfg.get('reduce_zero_label', False)

        if isinstance(self.img_scale, list):
            self.img_scale = tuple(self.img_scale)

        if id_path is None:
            id_path = f'splits/{self.name}/pred.txt' if mode == 'pred' else f'splits/{self.name}/val.txt'
        with open(id_path, 'r') as f:
            self.ids = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    def __len__(self):
        return len(self.ids)

    def _to_path(self, p):
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    def __getitem__(self, idx):
        id_str = self.ids[idx].strip()
        parts = id_str.split(' ')
        assert len(parts) >= 1, f'Bad line: "{id_str}"'

        img_path = self._to_path(parts[0])
        img = Image.open(img_path).convert('RGB')

        if len(parts) >= 2:
            mask_path = self._to_path(parts[1])
            mask_img = Image.fromarray(np.array(Image.open(mask_path)))
            if self.reduce_zero_label:
                m = np.array(mask_img)
                m[m == 0] = 255
                m = m - 1
                m[m == 254] = 255
                mask_img = Image.fromarray(m)
        else:
            w, h = img.size
            mask_img = Image.fromarray(np.zeros((h, w), dtype=np.uint8))

        if self.img_scale is not None:
            res = Resize(img_scale=self.img_scale, min_size=512)(dict(img=np.array(img)))
            img = Image.fromarray(res['img'])

        img_t, mask_t = normalize(img, mask_img)

        return img_t, mask_t, id_str
