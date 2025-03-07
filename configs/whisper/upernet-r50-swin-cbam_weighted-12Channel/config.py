crop_size = (
    256,
    256,
)
data_preprocessor = dict(
    bgr_to_rgb=False,
    mean=[
        1402.190971,
        1182.777247,
        1082.159965,
        1004.920854,
        1232.190289,
        1851.273666,
        2109.890887,
        2086.311509,
        2299.676992,
        1013.807055,
        1694.671876,
        1071.289589,
        -1513.374728,
        -826.21479,
        -1169.794759,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        256,
        256,
    ),
    std=[
        459.122149,
        505.267552,
        516.949052,
        627.047478,
        611.832499,
        578.15353,
        621.634638,
        635.887324,
        676.257714,
        536.378162,
        653.304738,
        514.812424,
        430.768388,
        377.389568,
        404.078978,
    ],
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
data_root = '/workspace/yejuni4196/whispers_2024/datasets/MMSeg-YREB/'
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
# load_from = '/workspace/yejuni4196/whispers_2024/datasets/mmsegmentation/weights/cbamr50_swin_normalize_256x256_upernet_80k_jhmin_cutout_ce_weight_all/iter_97500.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
lr_config = dict(
    _delete_=True,
    by_epoch=False,
    min_lr=0.0,
    policy='poly',
    power=1.0,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=9,
        num_convs=1,
        type='FCNHead'),
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
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            class_weight=[
                1.0,
                5.0,
                3.0,
                2.0,
                10.0,
                4.0,
                4.0,
                4.0,
                7.0,
            ],
            loss_weight=2.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=9,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='UPerHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    data_preprocessor=data_preprocessor,
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 9
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=3e-05,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(layer_decay_rate=0.9, num_layers=12),
    type='OptimWrapper')
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_20220317-55b0104a.pth'
randonness = dict(seed=42)
resume = False
sar_avg_mean = -1169.794759
sar_avg_std = 404.078978
test_cfg = dict(type='TestLoop')
test_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),

    dict(type='Resize', keep_ratio=True, scale=(256,256)),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='ConcatCDInput'),
    dict(type='PackSegInputs')
]
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(
            img_path='s2',
            img_path2='s1',
            seg_map_path='test/label'),
        data_root='/home/ssb/SEGMENTATION/DATA/whisper2024/MultiSenNA/Extracted/',
        # pipeline=[
        #     dict(type='LoadMultipleRSImageFromFile'),
        #     dict(keep_ratio=True, scale=(
        #         256,
        #         256,
        #     ), type='Resize'),
        #     dict(reduce_zero_label=True, type='LoadAnnotations'),
        #     dict(type='ConcatMultiInput'),
        #     dict(type='PackMultiInputs'),
        # ],
        type='YREBtinyDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ],
    keep_results=True,
    output_dir='outputs/cutout_ce_weight_all_97/',
    type='IoUMetric')
train_cfg = dict(
    max_iters=320000, type='IterBasedTrainLoop', val_interval=2500)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(
            img_path='train/MSI',
            img_path2='train/SAR_AVG_TIF',
            seg_map_path='train/label'),
        data_root='/workspace/yejuni4196/whispers_2024/datasets/MMSeg-YREB/',
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
                crop_size=(
                    256,
                    256,
                ),
                type='MultiImgRandomCrop'),
            dict(prob=0.5, type='MultiImgRandomFlip'),
            dict(alpha=0.7, prob=0.5, type='MultiImgCutOut'),
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
            img_path='val/MSI',
            img_path2='val/SAR_AVG_TIF',
            seg_map_path='val/label'),
        data_root='/workspace/yejuni4196/whispers_2024/datasets/MMSeg-YREB/',
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
        'mIoU',
        'mFscore',
    ], type='IoUMetric')
vh_mean = -826.21479
vh_std = 377.389568
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
vv_mean = -1513.374728
vv_std = 430.768388
work_dir = '/workspace/yejuni4196/whispers_2024/datasets/mmsegmentation/weights/cbamr50_swin_normalize_256x256_upernet_80k_jhmin_cutout_ce_weight_all'
