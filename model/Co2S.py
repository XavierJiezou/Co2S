import numpy as np
import torch
import torch.nn.functional as F
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

from model.text_embeddings import (
    aggregate_concept_predictions,
    get_class_to_concept_idxs,
)
from model.learnable_queries import LearnableQueries


def collect_4d(feats):
    out = []
    if isinstance(feats, torch.Tensor) and feats.dim() == 4:
        out.append(feats)
    elif isinstance(feats, (list, tuple)):
        for e in feats:
            out.extend(collect_4d(e))
    return out


def gather_4d_maps(raw):
    maps = []
    if isinstance(raw, torch.Tensor) and raw.dim() == 4:
        maps.append(raw)
    elif isinstance(raw, (list, tuple)):
        for e in raw:
            maps.extend(gather_4d_maps(e))
    return maps


def expected_c1_in_channels_from_head(head) -> int:
    if hasattr(head, "c1_proj") and isinstance(head.c1_proj, torch.nn.Sequential):
        conv = head.c1_proj[0]
        if isinstance(conv, torch.nn.Conv2d):
            return conv.in_channels
    if hasattr(head, "c1_proj") and hasattr(head.c1_proj[0], "weight"):
        return head.c1_proj[0].weight.shape[1]
    raise RuntimeError("Cannot infer c1 in_channels from decode_head.c1_proj")


def expected_c4_in_channels_from_head(head) -> int:
    if hasattr(head, "in_channels"):
        return int(head.in_channels)
    raise RuntimeError("Cannot infer ASPP in_channels from decode_head.in_channels")


def select_c1_c4(maps, head):
    exp_c1 = expected_c1_in_channels_from_head(head)
    exp_c4 = expected_c4_in_channels_from_head(head)

    c1_idx = None
    for i, m in enumerate(maps):
        if m.dim() == 4 and m.shape[1] == exp_c1:
            c1_idx = i
            break

    c4_idx = None
    for i in range(len(maps) - 1, -1, -1):
        m = maps[i]
        if m.dim() == 4 and m.shape[1] == exp_c4:
            c4_idx = i
            break

    if c1_idx is None or c4_idx is None:
        if len(maps) >= 2:
            c1, c4 = maps[-2], maps[-1]
        else:
            c1 = c4 = maps[-1]
    else:
        c1, c4 = maps[c1_idx], maps[c4_idx]

    if c1.shape[1] != exp_c1 and c4.shape[1] == exp_c1:
        c1, c4 = c4, c1
    return c1, c4


@SEGMENTORS.register_module()
class Co2S(EncoderDecoder):
    def __init__(
        self,
        backbone_type="clip",   
        text_embedding=None,
        freeze_backbone=False,
        exclude_keys=None,
        load_text_embedding=None,
        load_mcc_text_embedding=None,
        load_pl_text_embedding=None,
        clip_encoder=None,
        conv_encoder=None,
        maskclip_class_filter=None,
        maskclip_trust_head=None,
        renorm_clip_img=False,
        **args,
    ):
        super(Co2S, self).__init__(**args)

        self.backbone_type = backbone_type
        self.renorm_clip_img = renorm_clip_img

        self.clip_encoder = builder.build_backbone(clip_encoder) if clip_encoder is not None else None
        self.conv_encoder = builder.build_backbone(conv_encoder) if conv_encoder is not None else None

        self.text_provider = None
        self._cached_text = None

        requires_text = self.head_requires_text()

        if (text_embedding is None) and (load_text_embedding is None):
            if not requires_text:
                pass
            else:
                raise ValueError("Please provide `text_embedding` or `load_text_embedding` for text-aware heads.")
        else:
            te = text_embedding or dict(type="clip_npy", path=load_text_embedding)
            te_type = te.get("type", "").lower()

            if te_type in ("clip_npy", "npy"):
                npy_path = te.get("path", None)
                if npy_path is None:
                    raise ValueError("text_embedding.path is required for type='clip_npy'/'npy'.")
                arr = np.load(npy_path).astype(np.float32)  # [N, C]
                self._cached_text = torch.nn.Parameter(torch.from_numpy(arr), requires_grad=False)
                if hasattr(self.decode_head, "text_in_channels"):
                    assert self.decode_head.text_in_channels == arr.shape[1], \
                        f"text_in_channels={getattr(self.decode_head,'text_in_channels',None)} != loaded dim={arr.shape[1]}"
                if hasattr(self.decode_head, "load_text_embedding"):
                    self.decode_head.load_text_embedding = npy_path

            elif te_type == "simple":
                d_text = te.get("d_text", getattr(self.decode_head, "text_in_channels", 256))
                freeze = te.get("freeze", False)
                if hasattr(self.decode_head, "text_in_channels"):
                    assert self.decode_head.text_in_channels == d_text, \
                        f"text_in_channels={getattr(self.decode_head,'text_in_channels',None)} != d_text={d_text}"
                self.text_provider = LearnableQueries(
                    num_classes=self.decode_head.num_classes,
                    d_text=d_text,
                    freeze=freeze
                )
            else:
                raise ValueError(f"Unknown text_embedding type: {te_type}. expected 'clip_npy'/'npy' or 'simple'.")

        self.load_mcc_text_embedding = load_mcc_text_embedding
        self.loaded_mcc_text_feat = None
        if self.load_mcc_text_embedding:
            self.loaded_mcc_text_feat = torch.from_numpy(
                np.load(self.load_mcc_text_embedding).astype(np.float32)
            )

        if freeze_backbone:
            self.freeze(self.backbone, exclude_keys=exclude_keys)

    def head_requires_text(self) -> bool:
        hd = self.decode_head
        flag = getattr(hd, "expects_text", None)
        if flag is not None:
            return bool(flag)
        return hasattr(hd, "text_in_channels")


    def renormalize_img_for_clip(self, img: torch.Tensor) -> torch.Tensor:
        if not self.renorm_clip_img:
            return img
        loader_mean, loader_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        clip_mean, clip_std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        loader_mean = torch.tensor(loader_mean, device=img.device).view(1, -1, 1, 1)
        loader_std = torch.tensor(loader_std, device=img.device).view(1, -1, 1, 1)
        clip_mean = torch.tensor(clip_mean, device=img.device).view(1, -1, 1, 1)
        clip_std = torch.tensor(clip_std, device=img.device).view(1, -1, 1, 1)
        return (img * loader_std + loader_mean - clip_mean) / clip_std

    def freeze(self, model, exclude_keys=None):
        for n, m in model.named_parameters():
            m.requires_grad = False
            if exclude_keys is not None:
                assert isinstance(exclude_keys, list)
                for k in exclude_keys:
                    if str(k) in n:
                        m.requires_grad = True

    def forward_maskclip(self, img: torch.Tensor, conf_tresh: float):
        if self.clip_encoder is None or self.loaded_mcc_text_feat is None:
            raise NotImplementedError("forward_maskclip requires clip_encoder and load_mcc_text_embedding.")
        img = self.renormalize_img_for_clip(img)
        self.clip_encoder.eval()
        with torch.no_grad():
            text_feat = self.loaded_mcc_text_feat.detach().to(img.device)  # [N, C]
            visual_feat, _ = self.clip_encoder(img)
            visual_feat = visual_feat[-1]  # [B, C, H, W]
            dense_pred = F.conv2d(visual_feat, text_feat[:, :, None, None])  # [B, N, H, W]
            if dense_pred.shape[1] != self.num_classes:
                cls2con = get_class_to_concept_idxs(self.load_mcc_text_embedding)
                dense_pred = aggregate_concept_predictions(dense_pred, cls2con)
            assert dense_pred.shape[1] == self.num_classes
            dense_pred = F.interpolate(
                dense_pred, size=img.shape[-2:], mode="bilinear", align_corners=self.decode_head.align_corners
            )
            dense_pred = (100.0 * dense_pred).softmax(dim=1)
            dense_pred_certainty, dense_pred = dense_pred.max(dim=1)
            filtered_dense_pred = dense_pred.clone()
            filtered_dense_pred[dense_pred_certainty < conf_tresh] = 255
        return filtered_dense_pred


    def extract_feat(self, img: torch.Tensor):
        orig_img = img

        if self.backbone_type == "clip":
            img = self.renormalize_img_for_clip(img)

        requires_text = self.head_requires_text()

        if self.backbone_type == "dinov3":
            maps = []
            if hasattr(self.backbone, "get_intermediate_layers"):
                outs = self.backbone.get_intermediate_layers(img, n=[2, 7, 11], reshape=True, norm=True)
                if isinstance(outs, tuple):
                    outs = list(outs)
                maps = [o for o in outs if isinstance(o, torch.Tensor) and o.dim() == 4]
            if len(maps) == 0:  # fallback
                raw = self.backbone(img)
                maps = gather_4d_maps(raw)

        else:
            # other backbones return 4D maps directly or nested
            raw = self.backbone(img)
            maps = gather_4d_maps(raw)

        if len(maps) == 0:
            raise RuntimeError("No 4D feature maps extracted from backbone outputs.")

        if requires_text:
            pyramid = maps
            visual_feat = [pyramid]
            text_feat = None
            if isinstance(self.text_provider, LearnableQueries):
                text_feat = self.text_provider(img.shape[0])  # [B, N, C]
            elif self._cached_text is not None:
                text_feat = self._cached_text.to(img.device).unsqueeze(0).expand(img.shape[0], -1, -1)
            else:
                raise RuntimeError("Text head requires text embeddings, but none provided.")
            text_feat = torch.nan_to_num(text_feat, nan=0.0, posinf=1e4, neginf=-1e4)
        else:
            c1, c4 = select_c1_c4(maps, self.decode_head)
            visual_feat = [[(c1, c4)]]
            text_feat = None

        conv_feat = self.conv_encoder(orig_img) if self.conv_encoder is not None else None
        return [visual_feat, text_feat, conv_feat]

    # ---------- decode  ----------
    def _decode_head_forward_test(self, x, img_metas):
        if self.head_requires_text():
            out = self.decode_head.forward(x, force_output_pred_masks=True)
            return out["pred_masks"]

        vis = x[0]

        def _flatten_4d(nested):
            flat = []
            def rec(y):
                if isinstance(y, (list, tuple)):
                    for z in y:
                        rec(z)
                elif isinstance(y, torch.Tensor) and y.dim() == 4:
                    flat.append(y)
            rec(nested)
            return flat

        c1_c4 = None
        if isinstance(vis, (list, tuple)):
            # case A: [[(c1,c4)]]
            if len(vis) == 1 and isinstance(vis[0], (list, tuple)) and len(vis[0]) == 2 \
               and all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in vis[0]):
                c1_c4 = (vis[0][0], vis[0][1])
            # case B: [(c1,c4)]
            elif len(vis) == 2 and all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in vis):
                c1_c4 = (vis[0], vis[1])
            # case C: [c1, ..., c4]
            elif all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in vis) and len(vis) >= 2:
                c1_c4 = (vis[0], vis[-1])
            else:
                flat = _flatten_4d(vis)
                if len(flat) >= 2:
                    c1_c4 = (flat[0], flat[-1])

        if c1_c4 is None:
            raise RuntimeError("Cannot normalize visual features to (c1, c4) for decode head.")

        seg_logits = self.decode_head.forward(c1_c4)
        if isinstance(seg_logits, dict): 
            seg_logits = seg_logits.get("pred", seg_logits.get("pred_masks"))
        return seg_logits
