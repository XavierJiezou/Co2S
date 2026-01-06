import argparse
import logging
import os
import yaml
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.builder import build_model
from third_party.unimatch.dataset.semi_pred import SemiDataset
from datasets.palettes import get_palette
from third_party.unimatch.supervised_pred import predict
from third_party.unimatch.util.dist_helper import setup_distributed
from third_party.unimatch.util.utils import init_log


def _load_cfg(cfg_path: str):
    with open(cfg_path, "r") as fp:
        return yaml.load(fp, Loader=yaml.Loader)


def _override_model_from_flag(cfg: dict, model_flag: str):
    if not model_flag:
        return cfg
    mapping = {
        "clip": "mmseg.dual_model_clip",
        "dinov3": "mmseg.dual_model_dinov3",
    }
    if model_flag not in mapping:
        raise ValueError(f"--model must be 'clip' or 'dinov3', received: {model_flag}")
    cfg["model"] = mapping[model_flag]
    return cfg


def _load_checkpoint(model, ckpt_path: str, logger):
    ckpt = torch.load(ckpt_path, map_location="cuda")
    state = None
    for k in ("model", "state_dict", "state"):
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            state = ckpt[k]; break
    if state is None and isinstance(ckpt, dict):
        state = ckpt
    if state is None:
        raise RuntimeError(f"Bad checkpoint format: {type(ckpt)} keys={list(ckpt.keys()) if isinstance(ckpt, dict) else 'NA'}")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    logger.info(f"Loaded weights: {ckpt_path}")


@torch.no_grad()
def infer(model, dataloader, cfg, output_dir, distributed=False):
    model.eval()
    try:
        palette = get_palette(cfg["dataset"])
    except:
        palette = [i for i in range(256*3)]
        
    os.makedirs(output_dir, exist_ok=True)

    if distributed and torch.distributed.get_rank() == 0:
        print(f"Saving results to: {output_dir}")

    for data in tqdm(dataloader, total=len(dataloader), disable=(distributed and torch.distributed.get_rank() != 0)):
        img = data[0].cuda()
        id_str = data[2][0]

        file_name = os.path.basename(id_str)
        base_name = os.path.splitext(file_name)[0]
        
        pred_file = os.path.join(output_dir, f"{base_name}.png")

        pred = predict(
            model,
            img,
            mask=None,
            mode=cfg.get("eval_mode", "sliding_window"),
            cfg=cfg,
            return_logits=False,
        )

        os.makedirs(os.path.dirname(pred_file), exist_ok=True)
        np_pred = pred[0].cpu().numpy().astype(np.uint8)
        output = Image.fromarray(np_pred).convert('P')
        output.putpalette(palette)
        output.save(pred_file)

    if distributed and torch.distributed.get_rank() == 0:
        print(f"Inference completed.")


def main():
    parser = argparse.ArgumentParser(description="Co2S Pure Inference")
    
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--pred-path', type=str, required=True, help='Path to text file containing image list')
    
    parser.add_argument('--model', type=str, choices=['clip', 'dinov3'], default=None, 
                        help="Override model type")
    parser.add_argument('--output-dir', type=str, default='inference_results', 
                        help='Directory to save results (default: inference_results)')
    
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)

    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    cfg = _override_model_from_flag(cfg, args.model)
    
    cfg["mode"] = "pred"

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    rank, world_size = setup_distributed(port=args.port) if args.port is not None else (0, 1)
    
    model = build_model(cfg)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    _load_checkpoint(model, args.checkpoint, logger)

    if args.port is not None or 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            output_device=local_rank,
            find_unused_parameters=False
        )

    dataset = SemiDataset(cfg, 'pred', id_path=args.pred_path)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if (args.port is not None or 'LOCAL_RANK' in os.environ) else None
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4,
                            drop_last=False, sampler=sampler)

    infer(model, dataloader, cfg, args.output_dir, distributed=(rank > 0 or world_size > 1))


if __name__ == '__main__':
    main()