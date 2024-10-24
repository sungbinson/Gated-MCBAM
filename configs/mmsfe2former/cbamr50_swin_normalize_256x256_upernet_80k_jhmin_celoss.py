vv_mean, vh_mean = -827.528197, -451.801359
vv_std, vh_std = 817.995199, 497.044582
find_unused_parameters=True

sar_avg_mean = (vv_mean + vh_mean) / 2
sar_avg_std = (vv_std + vh_std) / 2

pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_20220317-55b0104a.pth'
crop_size = (256,256)
norm_cfg = dict(type='SyncBN', requires_grad=True)
randonness=dict(seed=42)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[803.731528, 857.797532, 815.581758, 1108.469372, 1866.404542, 2144.363235,
        2297.024816, 2257.105241, 1728.549177, 1051.305778, vv_mean, vh_mean, sar_avg_mean],
    std=[596.663639, 518.037177, 600.899631, 567.738704, 687.133744, 792.355984,
        841.339329, 848.200523, 694.178807, 546.315579, vv_std, vh_std, sar_avg_std],
    size=crop_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

# vv_mean , vh_mean = -1513.374728 , -826.214790
# vv_std , vh_std = 430.768388, 377.389568
# # SAR AVG
# sar_avg_mean = -1169.794759
# sar_avg_std = 404.078978
# pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_20220317-55b0104a.pth'
# crop_size = (256,256)
# norm_cfg = dict(type='SyncBN', requires_grad=True)

# # randonness=dict(seed=42)

# data_preprocessor = dict(
#     type='SegDataPreProcessor',
#     mean=[1402.190971, 1182.777247, 1082.159965, 1004.920854, 1232.190289, 1851.273666,
#         2109.890887, 2086.311509, 2299.676992, 1013.807055, 1694.671876, 1071.289589, vv_mean, vh_mean,sar_avg_mean ],
#     std=[459.122149, 505.267552, 516.949052, 627.047478, 611.832499, 578.153530,
#         621.634638, 635.887324, 676.257714, 536.378162, 653.304738, 514.812424, vv_std, vh_std, sar_avg_std],
#     bgr_to_rgb=False,
#     pad_val=0,
#     seg_pad_val=255,
#     size=crop_size,
#     test_cfg=dict(size_divisor=32)
# )



data_root  = '/home/hjkim/seg-challenge/MMSeg-YREB_last'
# data_root  = '/home/hjkim/seg-challenge/Dataset/MMSeg-YREB_v5'
dataset_type = 'YREBtinyDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=2500, save_best='mIoU',
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

launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    data_preprocessor = data_preprocessor,
    backbone=dict(
        swin_attn_drop_rate=0.0,
        swin_depths=[
            2,
            2,
            18,
            2,
        ],
        swin_drop_path_rate=0.3,
        swin_drop_rate=0.0,
        swin_embed_dims=128,
        swin_frozen_stages=-1,
        swin_in_channel=12,
        swin_init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_20220317-55b0104a.pth',
            type='Pretrained'),
        swin_mlp_ratio=4,
        swin_num_heads=[
            4,
            8,
            16,
            32,
        ],
        swin_out_indices=(
            0,
            1,
            2,
            3,
        ),
        swin_patch_norm=True,
        swin_pretrain_img_size=384,
        swin_qk_scale=None,
        swin_qkv_bias=True,
        swin_window_size=12,
        swin_with_cp=False,
        type='MMEncoder_v3'),

        # neck=dict(
        # type='Feature2Pyramid',
        # in_channels=[256, 512, 1024, 2048],  # backbone의 출력 채널과 일치
        # out_channels=[256, 512, 1024, 2048],  # 각 레벨의 출력 채널 유지
        # rescales=[4, 2, 1, 0.5],
        # norm_cfg=dict(type='SyncBN', requires_grad=True)
        # ),
        
        decode_head=dict(
        type='UPerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0) ),
        # loss_decode=dict(
        #     type='FocalLoss', use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     reduction='mean', loss_weight=0.4) ),

        auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,  
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=9,  
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        # loss_decode=dict(
        #     type='FocalLoss', use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     reduction='mean', loss_weight=0.4) 

            ),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole'),
        type='EncoderDecoder')


num_classes = 9
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=3e-5,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9),
    type='OptimWrapper')
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='val/multisen',
            img_path2='val/SAR_AVG_TIF',
            seg_map_path='val/label'),
        data_root=data_root,
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

train_cfg = dict(
    max_iters=320000, type='IterBasedTrainLoop', val_interval=2500)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(
            img_path='train/multisen',
            img_path2='train/SAR_AVG_TIF',
            seg_map_path='train/label'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadMultipleRSImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                ratio_range=(
                    0.5,
                    2,
                ),
                scale=(
                    256,
                    256,
                ),
                type='MultiImgRandomResize'),
            dict(
                cat_max_ratio=0.75,
                crop_size=
                    crop_size,
                type='MultiImgRandomCrop'),
            dict(prob=0.5, type='MultiImgRandomFlip'),
            dict(type='MultiImgCutOut', prob=0.5, alpha=0.7),

            dict(type='ConcatMultiInput'),
            dict(type='PackMultiInputs'),
        ],
        type='YREBtinyDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))

tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='val/multisen',
            img_path2='val/SAR_AVG_TIF',
            seg_map_path='val/label'),
        data_root=data_root,
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
val_evaluator = dict(
    iou_metrics=[
        'mIoU', 'mFscore'
    ], type='IoUMetric')


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])

