import os
import sys
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)

from third_party.maskclip.models.backbones.maskclip_vit import MaskClipVisionTransformer
from third_party.dinov3.models.vision_transformer import DinoVisionTransformer

# -------------------------------------------------------------------------
# Model Builders
# -------------------------------------------------------------------------

def build_clip(ckpt_path, device):
    """Build CLIP ViT-B/16 (MaskClip Version)."""
    model = MaskClipVisionTransformer(
        img_size=224, patch_size=16,
        embed_dims=768, num_layers=12, num_heads=12,
        mlp_ratio=4, out_indices=(11,),
        with_cls_token=True, return_clip_embed=False,
        pre_norm=True, final_norm=True, patch_bias=False,
        return_qkv=True
    ).to(device)
    _load_checkpoint(model, ckpt_path)
    return model

def build_dinov3(ckpt_path, device):
    """Build DINOv3 ViT-B/16."""
    model = DinoVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        layerscale_init=1e-5, norm_layer="layernorm",
        ffn_layer="mlp", ffn_bias=True, proj_bias=True,
        n_storage_tokens=4, mask_k_bias=True
    ).to(device)
    _load_checkpoint(model, ckpt_path)
    return model

def _load_checkpoint(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # Remove generic backbone prefix if present
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False) # strict=False for CLIP due to proj layers
    model.eval()

# -------------------------------------------------------------------------
# Preprocessing & Utils
# -------------------------------------------------------------------------

def preprocess_image(path):
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at: {path}")
    pil_img = Image.open(path).convert("RGB")
    return tfm(pil_img).unsqueeze(0), pil_img

def save_single_map(save_path, background_img, overlay_map=None, alpha=0.6, cmap='jet'):
    """Save a single image with optional heatmap overlay, no axis."""
    fig = plt.figure(figsize=(4, 4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    ax.imshow(background_img)
    if overlay_map is not None:
        ax.imshow(overlay_map, cmap=cmap, alpha=alpha)
        
    plt.savefig(save_path, dpi=100)
    plt.close()

# -------------------------------------------------------------------------
# Extraction Logic
# -------------------------------------------------------------------------

def extract_clip_attention(model, x):
    """Run CLIP and extract attention maps (Average, Max, Heads)."""
    with torch.no_grad():
        out, q, k, v = model(x)[0]
    
    B, N, C = q.shape
    num_heads = 12
    head_dim = C // num_heads

    # Reshape: [B, N, 12, 64] -> [B, 12, N, 64]
    q_heads = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    k_heads = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

    # Attention Matrix
    scale = head_dim ** -0.5
    attn_matrix = (q_heads @ k_heads.transpose(-2, -1)) * scale
    attn_matrix = attn_matrix.softmax(dim=-1)

    # Global Saliency (mean over query dim since CLS is missing)
    head_saliency = attn_matrix.mean(dim=2) 
    
    return _process_maps(head_saliency, N)

def extract_dinov3_attention(model, x):
    """Run DINOv3 and extract attention maps via hooks."""
    qkv_storage = {}
    def qkv_hook(module, input, output):
        qkv_storage['out'] = output
    
    target_layer = model.blocks[-1].attn.qkv
    handle = target_layer.register_forward_hook(qkv_hook)
    
    with torch.no_grad():
        model(x)
    handle.remove()
    
    # Process QKV
    qkv = qkv_storage['out']
    B, N_all, three_dim = qkv.shape
    num_heads = model.num_heads
    head_dim = (three_dim // 3) // num_heads
    
    qkv = qkv.reshape(B, N_all, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    
    scale = head_dim ** -0.5
    attn_weights = (q @ k.transpose(-2, -1)) * scale
    attn_weights = attn_weights.softmax(dim=-1)
    
    # Slice: Skip CLS + Registers
    num_registers = model.n_storage_tokens
    cls_start_idx = 1 + num_registers
    cls_attn = attn_weights[:, :, 0, cls_start_idx:] # [B, Heads, N_patches]
    
    return _process_maps(cls_attn, cls_attn.shape[-1])

def _process_maps(head_tensor, num_patches):
    """Helper to upscale and normalize maps."""
    h = w = int(num_patches ** 0.5)
    num_heads = head_tensor.shape[1]
    
    head_maps = []
    # 1. Process Individual Heads
    for i in range(num_heads):
        raw_map = head_tensor[0, i].reshape(h, w).cpu().numpy()
        norm_map = (raw_map - raw_map.min()) / (raw_map.max() - raw_map.min() + 1e-6)
        
        t_map = torch.tensor(norm_map).unsqueeze(0).unsqueeze(0)
        t_map = F.interpolate(t_map, size=(224, 224), mode="bicubic", align_corners=False)
        head_maps.append(t_map[0, 0].numpy())

    # 2. Average Map
    avg_raw = head_tensor.mean(dim=1)[0].reshape(h, w).cpu().numpy()
    avg_map = (avg_raw - avg_raw.min()) / (avg_raw.max() - avg_raw.min() + 1e-6)
    t_avg = torch.tensor(avg_map).unsqueeze(0).unsqueeze(0)
    t_avg = F.interpolate(t_avg, size=(224, 224), mode="bicubic", align_corners=False)
    final_avg = t_avg[0, 0].numpy()

    # 3. Max Map
    max_raw = head_tensor.max(dim=1)[0][0].reshape(h, w).cpu().numpy()
    max_map = (max_raw - max_raw.min()) / (max_raw.max() - max_raw.min() + 1e-6)
    t_max = torch.tensor(max_map).unsqueeze(0).unsqueeze(0)
    t_max = F.interpolate(t_max, size=(224, 224), mode="bicubic", align_corners=False)
    final_max = t_max[0, 0].numpy()

    return final_avg, final_max, head_maps

# -------------------------------------------------------------------------
# Main Controller
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize Attention Maps for CLIP or DINOv3")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, choices=['clip', 'dinov3'], required=True, help="Model architecture")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output images")
    args = parser.parse_args()

    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # 2. Load Model & Data
    print(f"Loading {args.model} model from {args.checkpoint}...")
    if args.model == 'clip':
        model = build_clip(args.checkpoint, device)
    else:
        model = build_dinov3(args.checkpoint, device)

    print(f"Processing image: {args.image_path}")
    tensor_in, pil_img = preprocess_image(args.image_path)
    tensor_in = tensor_in.to(device)
    pil_img_resized = pil_img.resize((224, 224))

    # 3. Extract Attention
    if args.model == 'clip':
        avg_map, max_map, head_maps = extract_clip_attention(model, tensor_in)
    else:
        avg_map, max_map, head_maps = extract_dinov3_attention(model, tensor_in)

    # 4. Save Results
    print(f"Saving results to {args.output_dir}...")
    
    # Individual Images
    save_single_map(os.path.join(args.output_dir, "original.png"), pil_img_resized, None)
    save_single_map(os.path.join(args.output_dir, "average.png"), pil_img_resized, avg_map, cmap='jet')
    save_single_map(os.path.join(args.output_dir, "max.png"), pil_img_resized, max_map, cmap='jet')
    
    for i, h_map in enumerate(head_maps):
        save_single_map(os.path.join(args.output_dir, f"head_{i}.png"), pil_img_resized, h_map, cmap='jet')

    # Summary Grid PDF
    print("Generating summary grid PDF...")
    rows, cols = 3, 5
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)

    # Row 1
    ax = plt.subplot(rows, cols, 1)
    ax.imshow(pil_img_resized)
    ax.set_title("Original Image", fontsize=14, fontweight='bold')
    ax.axis("off")

    ax = plt.subplot(rows, cols, 2)
    ax.imshow(pil_img_resized)
    ax.imshow(avg_map, cmap="jet", alpha=0.6)
    ax.set_title("Average Attention\n(All Heads)", fontsize=14, fontweight='bold')
    ax.axis("off")

    ax = plt.subplot(rows, cols, 3)
    ax.imshow(pil_img_resized)
    ax.imshow(max_map, cmap="jet", alpha=0.6)
    ax.set_title("Max Attention\n(Any Head)", fontsize=14, fontweight='bold')
    ax.axis("off")

    # Heads 0-11
    for i, h_map in enumerate(head_maps):
        plot_idx = i + 4
        if plot_idx > rows * cols: break
        ax = plt.subplot(rows, cols, plot_idx)
        ax.imshow(pil_img_resized)
        ax.imshow(h_map, cmap="jet", alpha=0.6)
        ax.set_title(f"Head {i}", fontsize=12)
        ax.axis("off")

    pdf_path = os.path.join(args.output_dir, "summary_grid.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Success! Full summary saved at: {pdf_path}")

if __name__ == "__main__":
    main()