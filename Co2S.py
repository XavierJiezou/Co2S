import argparse
import logging
import math
import os
import pprint
import shutil
import uuid
import time
from typing import Literal

from datetime import datetime
from mmcv.utils import Config
import mmcv
import torch
import torch.backends.cudnn as cudnn
import yaml
from matplotlib import pyplot as plt
from mmseg.core import build_optimizer
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

from datasets.palettes import get_palette
from experiments import get_git_revision
from model.builder import build_model
from third_party.unimatch.supervised import evaluate
from third_party.unimatch.dataset.semi import SemiDataset
from datasets.classes import CLASSES
from third_party.unimatch.util.ohem import ProbOhemCrossEntropy2d
from third_party.unimatch.util.dist_helper import setup_distributed
from third_party.unimatch.util.utils import count_params, count_training_params, init_log

from utils.train_utils import (DictAverageMeter, confidence_weighted_loss,
                               cutmix_img_, cutmix_mask)
from version import __version__


def dynamic_weight(
    step: int,
    total_steps: int,
    *,
    up_ratio: float = 0.2,
    mode: Literal["linear", "cosine"] = "cosine",
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be a positive integer.")
    if not (0.0 <= up_ratio <= 1.0):
        raise ValueError("up_ratio must be within the interval [0,1].")
    if step < 0 or step >= total_steps:
        return 0.0

    peak = 0.5
    up_steps = min(max(int(round(total_steps * up_ratio)), 0), total_steps)

    def shape(x: float) -> float:
        if mode == "linear":
            return x
        elif mode == "cosine":
            return 0.5 * (1 - math.cos(math.pi * x))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if up_steps > 0 and step < up_steps:
        x = 1.0 if up_steps == 1 else step / (up_steps - 1)
        x = min(max(x, 0.0), 1.0)
        return float(peak * shape(x))
    else:
        return float(peak)


def softmax_mse_loss_masked(student_logits, teacher_logits, mask):
    """
    student_logits, teacher_logits: [B, C, H, W]
    mask: [B, 1, H, W], where 1 indicates that this pixel participates in the stability loss.
    """
    student_prob = torch.softmax(student_logits, dim=1)
    teacher_prob = torch.softmax(teacher_logits, dim=1)

    diff2 = (student_prob - teacher_prob) ** 2  # [B, C, H, W]
    diff2 = diff2 * mask                        # Only keep the pixels with mask=1
    
    denom = mask.sum() * student_logits.shape[1]
    if denom == 0:
        return student_logits.new_tensor(0.0)

    return diff2.sum() / denom


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    parser.add_argument("--model_2", required=True, help="Config for the second model (DINOv3)")

    args = parser.parse_args()

    with open(args.config, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)
    labeled_id_path = f'splits/{cfg["dataset"]}/{cfg["split"]}/labeled.txt'
    unlabeled_id_path = f'splits/{cfg["dataset"]}/{cfg["split"]}/unlabeled.txt'

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    mmcv.utils.get_logger('mmcv').setLevel('WARNING')

    rank, world_size = setup_distributed(port=args.port)
    if cfg['nccl_p2p_disable']:
        os.environ["NCCL_P2P_DISABLE"] = str(1)

    if rank == 0:
        timestr = datetime.now().strftime("%y%m%d-%H%M")
        uid = str(uuid.uuid4())[:5]
        run_name = f'{timestr}_{cfg["name"]}_v{__version__}_{uid}'.replace('.', '-')
        save_path = f'exp/exp-{cfg["exp"]}/{run_name}'
        os.makedirs(save_path, exist_ok=True)

        formatter = logging.Formatter(fmt='[%(asctime)s] [%(levelname)-8s] %(message)s')
        fileHandler = logging.FileHandler(f'{save_path}/debug.log')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        all_args = {**cfg, **vars(args),
                    'labeled_id_path': labeled_id_path, 'unlabeled_id_path': unlabeled_id_path,
                    'ngpus': world_size, 'run_name': run_name, 'save_path': save_path,
                    'exec_git_rev': get_git_revision(), 'exec_version': __version__}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(save_path)

        shutil.copyfile(args.config, os.path.join(save_path, 'config.yaml'))
        with open(os.path.join(save_path, 'all_args.yaml'), 'w') as f:
            yaml.dump(all_args, f, default_flow_style=None, sort_keys=False, indent=2)

    cudnn.enabled = True
    cudnn.benchmark = True

    assert cfg['use_fp']
    assert cfg['pleval']

    # ================= Build models =================
    # Model 1: CLIP
    model_clip = build_model(cfg)
    
    # Model 2: DINOv3
    model_dinov3 = build_model({
        "model": args.model_2,
        "dataset": cfg["dataset"],
        "crop_size": cfg["crop_size"],
        "disable_dropout": cfg["disable_dropout"],
        "fp_rate": cfg["fp_rate"]
    })

    # ================= Optimizers =================
    if 'optimizer' not in cfg:
        # Default SGD for CLIP
        optimizer_clip = SGD([
            {'params': model_clip.backbone.parameters(), 'lr': cfg['lr']},
            {'params': [p for n, p in model_clip.named_parameters() if 'backbone' not in n],
             'lr': cfg['lr'] * cfg['lr_multi']}
        ], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

        # Default SGD for DINOv3
        optimizer_dinov3 = SGD([
            {'params': model_dinov3.backbone.parameters(), 'lr': cfg['lr']},
            {'params': [p for n, p in model_dinov3.named_parameters() if 'backbone' not in n],
             'lr': cfg['lr'] * cfg['lr_multi']}
        ], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    else:
        # Custom Optimizer for CLIP
        optimizer_clip = build_optimizer(model_clip, cfg['optimizer'])
        for group in optimizer_clip.param_groups:
            group.setdefault('initial_lr', group['lr'])

        # Custom Optimizer for DINOv3
        optimizer_dinov3 = build_optimizer(model_dinov3, cfg['optimizer'])
        for group in optimizer_dinov3.param_groups:
            group.setdefault('initial_lr', group['lr'])

    if rank == 0:
        if hasattr(model_clip, 'backbone'):
            logger.info(f'CLIP Backbone params (training/total): {count_training_params(model_clip.backbone):.1f}M/{count_params(model_clip.backbone):.1f}M')
            logger.info(f'DINOv3 Backbone params (training/total): {count_training_params(model_dinov3.backbone):.1f}M/{count_params(model_dinov3.backbone):.1f}M')
        if hasattr(model_clip, 'decode_head'):
            logger.info(f'CLIP Decoder params (training/total): {count_training_params(model_clip.decode_head):.1f}M/{count_params(model_clip.decode_head):.1f}M')
            logger.info(f'DINOv3 Decoder params (training/total): {count_training_params(model_dinov3.decode_head):.1f}M/{count_params(model_dinov3.decode_head):.1f}M')

    local_rank = int(os.environ["LOCAL_RANK"])
    
    # DDP Setup - CLIP
    model_clip = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_clip)
    model_clip.cuda()
    model_clip = torch.nn.parallel.DistributedDataParallel(
        model_clip, device_ids=[local_rank], broadcast_buffers=False,
        output_device=local_rank, find_unused_parameters=True
    )

    # DDP Setup - DINOv3
    model_dinov3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_dinov3)
    model_dinov3.cuda()
    model_dinov3 = torch.nn.parallel.DistributedDataParallel(
        model_dinov3, device_ids=[local_rank], broadcast_buffers=False,
        output_device=local_rank, find_unused_parameters=True
    )

    # ================= Criterion =================
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'mmseg':
        criterion_l = None
    else:
        raise ValueError(cfg['criterion_u']['name'])

    if cfg['criterion_u'] == 'CELoss':
        criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    elif cfg['criterion_u'] == 'mmseg':
        criterion_u = None
    else:
        raise ValueError(cfg['criterion_u'])

    # ================= Datasets =================
    trainset_u = SemiDataset(cfg, 'train_u', id_path=unlabeled_id_path)
    trainset_l = SemiDataset(cfg, 'train_l', id_path=labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg, 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)
    palette = get_palette(cfg['dataset'])

    if cfg['iters'] is not None:
        assert cfg['epochs'] is None
        cfg['epochs'] = math.ceil(cfg['iters'] / len(trainloader_u))

    total_iters = len(trainloader_u) * cfg['epochs']
    scheduler_max_iters = cfg.get('scheduler_max_iters', total_iters)
    assert scheduler_max_iters >= total_iters
    if rank == 0:
        logger.info(f'Train for {cfg["epochs"]} epochs / {total_iters} iterations.')
    
    best_miou_clip = 0.0
    best_miou_dinov3 = 0.0
    epoch = -1

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Best mIoU [CLIP/DINOv3]: {:.2f}/{:.2f}'.format(
                epoch, optimizer_clip.param_groups[0]['lr'], best_miou_clip, best_miou_dinov3))

        log_avg = DictAverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_w, img_s1, img_s2, ignore_mask, mix1, mix2),
                (img_w_other, img_s1_other, img_s2_other, ignore_mask_other, _, _)) in enumerate(loader):
            t0 = time.time()
            iters = epoch * len(trainloader_u) + i
            img_x = img_x.cuda()
            img_s1 = img_s1.cuda()
            img_s2 = img_s2.cuda()
            mask_x = mask_x.cuda()
            img_w = img_w.cuda()
            ignore_mask = ignore_mask.cuda()
            mix1 = mix1.cuda()
            mix2 = mix2.cuda()
            img_w_other = img_w_other.cuda()
            img_s1_other = img_s1_other.cuda()
            img_s2_other = img_s2_other.cuda()
            ignore_mask_other = ignore_mask_other.cuda()

            # CutMix images
            cutmix_img_(img_s1, img_s1_other, mix1)
            cutmix_img_(img_s2, img_s2_other, mix2)

            # ================= Generate pseudo labels (No Grad) =================
            with torch.no_grad():
                # CLIP
                model_clip.eval()
                pred_clip_w_other = model_clip(img_w_other).detach()
                conf_clip_w_other, mask_clip_w_other = pred_clip_w_other.softmax(dim=1).max(dim=1)

                # DINOv3
                model_dinov3.eval()
                pred_dinov3_w_other = model_dinov3(img_w_other).detach()
                conf_dinov3_w_other, mask_dinov3_w_other = pred_dinov3_w_other.softmax(dim=1).max(dim=1)

            model_clip.train()
            model_dinov3.train()

            # ================= CLIP Forward & Loss =================
            preds_clip, preds_clip_fp = model_clip(torch.cat((img_x, img_w)), need_fp=True)
            pred_clip_x, pred_clip_w = preds_clip.chunk(2)
            _, pred_clip_w_fp = preds_clip_fp.chunk(2)

            pred_clip_s1, pred_clip_s2 = model_clip(torch.cat((img_s1, img_s2))).chunk(2)

            pred_clip_w = pred_clip_w.detach()
            conf_clip_w, mask_clip_w = pred_clip_w.softmax(dim=1).max(dim=1)

            # CLIP CutMix labels
            mask_clip_w_mixed1 = cutmix_mask(mask_clip_w, mask_clip_w_other, mix1)
            mask_clip_w_mixed2 = cutmix_mask(mask_clip_w, mask_clip_w_other, mix2)
            conf_clip_w_mixed1 = cutmix_mask(conf_clip_w, conf_clip_w_other, mix1)
            conf_clip_w_mixed2 = cutmix_mask(conf_clip_w, conf_clip_w_other, mix2)
            ignore_mask_mixed1 = cutmix_mask(ignore_mask, ignore_mask_other, mix1)
            ignore_mask_mixed2 = cutmix_mask(ignore_mask, ignore_mask_other, mix2)

            # 1. Supervised Loss (CLIP)
            if criterion_l is not None:
                loss_clip_x = criterion_l(pred_clip_x, mask_x)
            else:
                losses = model_clip.module.decode_head.loss_decode({'pred_masks': pred_clip_x}, mask_x)
                loss_clip_x, log_vars_x = model_clip.module._parse_losses(losses)

            # 2. FixMatch Loss (CLIP)
            if criterion_u is not None:
                loss_clip_s1 = criterion_u(pred_clip_s1, mask_clip_w_mixed1)
                loss_clip_s1 = confidence_weighted_loss(loss_clip_s1, conf_clip_w_mixed1, ignore_mask_mixed1, cfg)
            else:
                loss_clip_s1, _ = model_clip.module._parse_losses(
                    model_clip.module.decode_head.loss_decode({'pred_masks': pred_clip_s1}, mask_clip_w_mixed1))
                conf_ratio = ((conf_clip_w_mixed1 >= cfg['conf_thresh']) & (ignore_mask_mixed1 != 255)).sum().item() / \
                             (ignore_mask_mixed1 != 255).sum().item()
                loss_clip_s1 *= conf_ratio

            if criterion_u is not None:
                loss_clip_s2 = criterion_u(pred_clip_s2, mask_clip_w_mixed2)
                loss_clip_s2 = confidence_weighted_loss(loss_clip_s2, conf_clip_w_mixed2, ignore_mask_mixed2, cfg)
            else:
                loss_clip_s2, _ = model_clip.module._parse_losses(
                    model_clip.module.decode_head.loss_decode({'pred_masks': pred_clip_s2}, mask_clip_w_mixed2))
                conf_ratio = ((conf_clip_w_mixed2 >= cfg['conf_thresh']) & (ignore_mask_mixed2 != 255)).sum().item() / \
                             (ignore_mask_mixed2 != 255).sum().item()
                loss_clip_s2 *= conf_ratio

            # 3. Feature Perturbation Loss (CLIP)
            if criterion_u is not None:
                loss_clip_fp = criterion_u(pred_clip_w_fp, mask_clip_w)
                loss_clip_fp = confidence_weighted_loss(loss_clip_fp, conf_clip_w, ignore_mask, cfg)
            else:
                loss_clip_fp, _ = model_clip.module._parse_losses(
                    model_clip.module.decode_head.loss_decode({'pred_masks': pred_clip_w_fp}, mask_clip_w))
                conf_ratio = ((conf_clip_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                             (ignore_mask != 255).sum().item()
                loss_clip_fp *= conf_ratio

            loss_clip_total = (loss_clip_x + loss_clip_s1 * 0.25 + loss_clip_s2 * 0.25 + loss_clip_fp * 0.5) / 2.0

            # ================= DINOv3 Forward & Loss =================
            preds_dinov3, preds_dinov3_fp = model_dinov3(torch.cat((img_x, img_w)), need_fp=True)
            pred_dinov3_x, pred_dinov3_w = preds_dinov3.chunk(2)
            _, pred_dinov3_w_fp = preds_dinov3_fp.chunk(2)

            pred_dinov3_s1, pred_dinov3_s2 = model_dinov3(torch.cat((img_s1, img_s2))).chunk(2)

            pred_dinov3_w = pred_dinov3_w.detach()
            conf_dinov3_w, mask_dinov3_w = pred_dinov3_w.softmax(dim=1).max(dim=1)

            # DINOv3 CutMix labels
            mask_dinov3_w_mixed1 = cutmix_mask(mask_dinov3_w, mask_dinov3_w_other, mix1)
            mask_dinov3_w_mixed2 = cutmix_mask(mask_dinov3_w, mask_dinov3_w_other, mix2)
            conf_dinov3_w_mixed1 = cutmix_mask(conf_dinov3_w, conf_dinov3_w_other, mix1)
            conf_dinov3_w_mixed2 = cutmix_mask(conf_dinov3_w, conf_dinov3_w_other, mix2)

            # 1. Supervised Loss (DINOv3)
            if criterion_l is not None:
                loss_dinov3_x = criterion_l(pred_dinov3_x, mask_x)
            else:
                losses_dinov3 = model_dinov3.module.decode_head.loss_decode({'pred_masks': pred_dinov3_x}, mask_x)
                loss_dinov3_x, log_vars_x_dinov3 = model_dinov3.module._parse_losses(losses_dinov3)

            # 2. FixMatch Loss (DINOv3)
            if criterion_u is not None:
                loss_dinov3_s1 = criterion_u(pred_dinov3_s1, mask_dinov3_w_mixed1)
                loss_dinov3_s1 = confidence_weighted_loss(loss_dinov3_s1, conf_dinov3_w_mixed1, ignore_mask_mixed1, cfg)
            else:
                loss_dinov3_s1, _ = model_dinov3.module._parse_losses(
                    model_dinov3.module.decode_head.loss_decode({'pred_masks': pred_dinov3_s1}, mask_dinov3_w_mixed1))
                conf_ratio_2 = ((conf_dinov3_w_mixed1 >= cfg['conf_thresh']) & (ignore_mask_mixed1 != 255)).sum().item() / \
                               (ignore_mask_mixed1 != 255).sum().item()
                loss_dinov3_s1 *= conf_ratio_2

            if criterion_u is not None:
                loss_dinov3_s2 = criterion_u(pred_dinov3_s2, mask_dinov3_w_mixed2)
                loss_dinov3_s2 = confidence_weighted_loss(loss_dinov3_s2, conf_dinov3_w_mixed2, ignore_mask_mixed2, cfg)
            else:
                loss_dinov3_s2, _ = model_dinov3.module._parse_losses(
                    model_dinov3.module.decode_head.loss_decode({'pred_masks': pred_dinov3_s2}, mask_dinov3_w_mixed2))
                conf_ratio_2 = ((conf_dinov3_w_mixed2 >= cfg['conf_thresh']) & (ignore_mask_mixed2 != 255)).sum().item() / \
                               (ignore_mask_mixed2 != 255).sum().item()
                loss_dinov3_s2 *= conf_ratio_2

            # 3. Feature Perturbation Loss (DINOv3)
            if criterion_u is not None:
                loss_dinov3_fp = criterion_u(pred_dinov3_w_fp, mask_dinov3_w)
                loss_dinov3_fp = confidence_weighted_loss(loss_dinov3_fp, conf_dinov3_w, ignore_mask, cfg)
            else:
                loss_dinov3_fp, _ = model_dinov3.module._parse_losses(
                    model_dinov3.module.decode_head.loss_decode({'pred_masks': pred_dinov3_w_fp}, mask_dinov3_w))
                conf_ratio_2 = ((conf_dinov3_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                               (ignore_mask != 255).sum().item()
                loss_dinov3_fp *= conf_ratio_2

            loss_dinov3_total = (loss_dinov3_x + loss_dinov3_s1 * 0.25 + loss_dinov3_s2 * 0.25 + loss_dinov3_fp * 0.5) / 2.0

            # ================= Stability / Mutual Learning =================
            with torch.no_grad():
                prob_clip   = torch.softmax(pred_clip_w,   dim=1)   # [B, C, H, W]
                prob_dinov3 = torch.softmax(pred_dinov3_w, dim=1)

                conf_clip,   pseudo_clip   = prob_clip.max(dim=1)      # [B, H, W]
                conf_dinov3, pseudo_dinov3 = prob_dinov3.max(dim=1)

                high_clip = conf_clip   >= cfg['conf_thresh']
                high_dino = conf_dinov3 >= cfg['conf_thresh']

                both_high      = high_clip & high_dino
                only_clip_high = high_clip & (~high_dino)
                only_dino_high = high_dino & (~high_clip)

                # DINOv3 teaches CLIP
                clip_student_mask = (only_dino_high | (both_high & (conf_dinov3 >= conf_clip)))   # [B, H, W]
                # CLIP teaches DINOv3
                dino_student_mask = (only_clip_high | (both_high & (conf_clip >= conf_dinov3)))   # [B, H, W]

            stabilization_weight = dynamic_weight(step=iters, total_steps=total_iters, up_ratio=0.2, mode="cosine")

            # CLIP is student: learns from DINOv3
            clip_student_mask_4d = clip_student_mask.unsqueeze(1).float()  # [B,1,H,W]
            loss_stab_clip = softmax_mse_loss_masked(
                student_logits=pred_clip_w,          # CLIP logits
                teacher_logits=pred_dinov3_w.detach(), # DINOv3 logits (detached)
                mask=clip_student_mask_4d
            ) * stabilization_weight

            # DINOv3 is student: learns from CLIP
            dino_student_mask_4d = dino_student_mask.unsqueeze(1).float()  # [B,1,H,W]
            loss_stab_dinov3 = softmax_mse_loss_masked(
                student_logits=pred_dinov3_w,        # DINOv3 logits
                teacher_logits=pred_clip_w.detach(), # CLIP logits (detached)
                mask=dino_student_mask_4d
            ) * stabilization_weight

            loss_clip_total   += loss_stab_clip
            loss_dinov3_total += loss_stab_dinov3

            # ================= Update =================
            optimizer_clip.zero_grad()
            loss_clip_total.backward()
            optimizer_clip.step()

            optimizer_dinov3.zero_grad()
            loss_dinov3_total.backward()
            optimizer_dinov3.step()

            # ================= LR Schedule =================
            if 'optimizer' not in cfg:
                if iters < cfg['warmup_iters']:
                    k = (1 - iters / cfg['warmup_iters']) * (1 - cfg['warmup_ratio'])
                    lr = cfg['lr'] * (1 - k)
                else:
                    lr = cfg['lr'] * (1 - iters / scheduler_max_iters) ** 0.9
                optimizer_clip.param_groups[0]["lr"] = lr
                optimizer_clip.param_groups[1]["lr"] = lr * cfg['lr_multi']

                optimizer_dinov3.param_groups[0]["lr"] = lr
                optimizer_dinov3.param_groups[1]["lr"] = lr * cfg['lr_multi']

            else:
                if iters < cfg['warmup_iters']:
                    k = (1 - iters / cfg['warmup_iters']) * (1 - cfg['warmup_ratio'])
                    for group in optimizer_clip.param_groups:
                        group['lr'] = group['initial_lr'] * (1 - k)
                    for group in optimizer_dinov3.param_groups:
                        group['lr'] = group['initial_lr'] * (1 - k)
                else:
                    for group in optimizer_clip.param_groups:
                        group['lr'] = group['initial_lr'] * (1 - iters / scheduler_max_iters) ** 0.9
                    for group in optimizer_dinov3.param_groups:
                        group['lr'] = group['initial_lr'] * (1 - iters / scheduler_max_iters) ** 0.9

            # ================= Logging =================
            log_avg.update({
                'train/iter_time': time.time() - t0,

                'train/clip/loss_all': loss_clip_total,
                'train/clip/loss_x': loss_clip_x,
                'train/clip/loss_s1': loss_clip_s1,
                'train/clip/loss_s2': loss_clip_s2,
                'train/clip/loss_fp': loss_clip_fp,
                'train/clip/stabilization_loss': loss_stab_clip,

                'train/dinov3/loss_all': loss_dinov3_total,
                'train/dinov3/loss_x': loss_dinov3_x,
                'train/dinov3/loss_s1': loss_dinov3_s1,
                'train/dinov3/loss_s2': loss_dinov3_s2,
                'train/dinov3/loss_fp': loss_dinov3_fp,
                'train/dinov3/stabilization_loss': loss_stab_dinov3,
            })

            if i % 100 == 0 and rank == 0:
                logger.info(f'Iters: {i} ' + str(log_avg))
                for k, v in log_avg.avgs.items():
                    writer.add_scalar(k, v, iters)
                log_avg.reset()

        # ================= Evaluation =================
        if epoch % cfg.get('eval_every_n_epochs', 1) == 0 or epoch == cfg['epochs'] - 1:
            eval_mode = cfg['eval_mode']
            mIoU_clip, iou_class_clip = evaluate(model_clip, valloader, eval_mode, cfg)
            mIoU_dinov3, iou_class_dinov3 = evaluate(model_dinov3, valloader, eval_mode, cfg)

            if rank == 0:
                logger.info(run_name)

                logger.info('***** Evaluation (CLIP) *****')
                for (cls_idx, iou) in enumerate(iou_class_clip):
                    logger.info('Class [{:} {:}] IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
                logger.info('MeanIoU (CLIP, {}): {:.2f}\n'.format(eval_mode, mIoU_clip))

                logger.info('***** Evaluation (DINOv3) *****')
                for (cls_idx, iou) in enumerate(iou_class_dinov3):
                    logger.info('Class [{:} {:}] IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
                logger.info('MeanIoU (DINOv3, {}): {:.2f}\n'.format(eval_mode, mIoU_dinov3))

                writer.add_scalar('eval/clip/mIoU', mIoU_clip, epoch)
                for i, iou in enumerate(iou_class_clip):
                    writer.add_scalar('eval/clip/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

                writer.add_scalar('eval/dinov3/mIoU', mIoU_dinov3, epoch)
                for i, iou in enumerate(iou_class_dinov3):
                    writer.add_scalar('eval/dinov3/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

            is_best_clip = mIoU_clip > best_miou_clip
            best_miou_clip = max(mIoU_clip, best_miou_clip)

            is_best_dinov3 = mIoU_dinov3 > best_miou_dinov3
            best_miou_dinov3 = max(mIoU_dinov3, best_miou_dinov3)

            if rank == 0:
                # Save CLIP
                checkpoint_clip = {
                    'model': model_clip.state_dict(),
                    'optimizer': optimizer_clip.state_dict(),
                    'epoch': epoch,
                }
                if is_best_clip:
                    torch.save(checkpoint_clip, os.path.join(save_path, 'best_clip.pth'))
                
                # Save DINOv3
                if is_best_dinov3:
                    checkpoint_dinov3 = {
                        'model': model_dinov3.state_dict(),
                        'optimizer': optimizer_dinov3.state_dict(),
                        'epoch': epoch,
                    }
                    torch.save(checkpoint_dinov3, os.path.join(save_path, 'best_dinov3.pth'))