norm_cfg = dict(type='SyncBN', requires_grad=True)
img_size = 512

model = dict(

    type='Co2SCLIPBEIT2',

    backbone=dict(
        type='BEiT2VisionTransformer',
        img_size=img_size,         
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_abs_pos_emb=False,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        out_indices=[3, 5, 7, 11],
        pretrained='pretrained/beitv2_base_patch16_224_pt1k_ft21ktoade20k.pth',
    ),

    backbone_type='beit2',

    text_embedding=dict(
        type='simple',
        d_text=768,
        normalize=True,
        freeze=False
    ),

    decode_head=dict(
        type='VLGHead',
        img_size=img_size,
        num_classes=6,               
        text_in_channels=768,
        text_channels=256,
        up_channels=(64, 32),
        skip_in_channels=(768, 768), 
        skip_channels=(32, 16),
        skip_from_conv_feat=False,
        num_layers=2,
        num_heads=4,
        channels=128,
        pool_size=(4, 4),
        conv1_ksize=7,
        align_corners=False,
        loss_decode=None,
    ),

    freeze_backbone=True,
    exclude_keys=['attn'],
)
