_base_ = [
    '../_base_/datasets/YREB.py',
]
dataset_type = 'YREBDataset'
data_root = '/home/whisper2024/MMSeg-YREB_v3/'
num_classes = 9
norm_cfg = dict(type='BN', requires_grad=True)

vv_mean , vh_mean = -1513.374728 , -826.214790
vv_std , vh_std = 430.768388, 377.389568
# SAR AVG
sar_avg_mean = -1169.794759
sar_avg_std = 404.078978

randomness=dict(seed=42)

crop_size = (256, 256)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[1402.190971, 1182.777247, 1082.159965 ,2109.890887, vv_mean, vh_mean,sar_avg_mean],
    std=[459.122149, 505.267552, 516.949052, 621.634638, vv_std, vh_std, sar_avg_std],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))
model = dict(
    data_preprocessor=data_preprocessor,
    type = 'EncoderDecoder',
    backbone=dict(
        swin_attn_drop_rate=0.0,
        swin_depths=[2,2,18,2,],
        swin_drop_path_rate=0.3,
        swin_drop_rate=0.0,
        swin_embed_dims=128,
        swin_frozen_stages=-1,
        swin_in_channel=4,
        swin_init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_20220317-55b0104a.pth',
            type='Pretrained'),
        swin_mlp_ratio=4,
        swin_num_heads=[4,8,16,32],
        swin_out_indices=(0,1,2,3,),
        swin_patch_norm=True,
        swin_pretrain_img_size=384,
        swin_qk_scale=None,
        swin_qkv_bias=True,
        swin_window_size=12,
        swin_with_cp=False,
        type='MMEncoder_v4'),

    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        sampler=dict(min_kept=100000, thresh=0.7, type='OHEMPixelSampler'),
        channels=1024,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=-1,
        channels=512,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_cfg = dict(
    max_iters=160000, type='IterBasedTrainLoop', val_interval=2500)


train_dataloader = dict(
   batch_size=16)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='val/RGB_NIR',
            img_path2='val/SAR_AVG_TIF',
            seg_map_path='val/label'),
        data_root= data_root,
        pipeline=[
            dict(type='LoadMultipleRSImageFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='ConcatMultiInput'),
            dict(type='PackMultiInputs'),
        ],
        type='YREBtinyDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')






default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=5000, save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=False, interval=1, type='SegVisualizationHook'))
default_scope = 'mmseg'
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]

# optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
optim_wrapper = dict(
    optimizer=dict(lr=3e-05, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')
# optimizer = dict(lr=3e-05, type='AdamW', weight_decay=0.05)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=40000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
"""
optimizer = dict(type='AdamW', lr=3e-5, weight_decay=0.05)
optim_wrapper = dict(

    type='OptimWrapper',
    optimizer=optimizer,
    )

param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=80000,
        eta_min=0,
        power=0.9,
        type='PolyLR'),
]
"""