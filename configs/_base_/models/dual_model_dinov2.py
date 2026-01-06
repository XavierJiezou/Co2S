norm_cfg = dict(type='SyncBN', requires_grad=True)
img_size = 256

model = dict(

    type='Co2SCLIPDINOV2',

    backbone=dict(
        type='Dino2VisionTransformer',   
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,            
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,

        pretrained='pretrained/dinov2_vitb16_256.pth',
    ),

    backbone_type="dinov2",

    decode_head=dict(
        type='DLV3PHead',
        img_size=img_size,
        in_channels=768,
        in_index=3,             
        channels=256,
        dilations=(6, 12, 18),
        c1_in_channels=768,
        c1_channels=48,
        dropout_ratio=0.0,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        init_cfg=None,
    ),

    freeze_backbone=True,
    exclude_keys=['attn'], 
)
