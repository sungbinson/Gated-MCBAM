conv_pretrained = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'
crop_size = (
    256,
    256,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        803.731528,
        857.797532,
        815.581758,
        1108.469372,
        1866.404542,
        2144.363235,
        2297.024816,
        2257.105241,
        1728.549177,
        1051.305778,
        -827.528197,
        -451.801359,
        -639.664778,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        256,
        256,
    ),
    std=[
        596.663639,
        518.037177,
        600.899631,
        567.738704,
        687.133744,
        792.355984,
        841.339329,
        848.200523,
        694.178807,
        546.315579,
        817.995199,
        497.044582,
        657.5198905,
    ],
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
data_root = '/home/hjkim/seg-challenge/MMSeg-YREB_last'
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
find_unused_parameters = True
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'pytorch'
load_from = '/home/hjkim/seg-challenge/mmsegmentation/workdir/whisper/Gcbamr50_conv_normalize_256x256_upernet_last_pretrained/10ch_best.pth'
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
        convn_arch='base',
        convn_drop_path_rate=0.4,
        convn_gap_before_final_norm=False,
        convn_init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth',
            prefix='backbone.',
            type='Pretrained'),
        convn_layer_scale_init_value=1.0,
        convn_out_indices=[
            0,
            1,
            2,
            3,
        ],
        type='MMEncoder_v2'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            803.731528,
            857.797532,
            815.581758,
            1108.469372,
            1866.404542,
            2144.363235,
            2297.024816,
            2257.105241,
            1728.549177,
            1051.305778,
            -827.528197,
            -451.801359,
            -639.664778,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            256,
        ),
        std=[
            596.663639,
            518.037177,
            600.899631,
            567.738704,
            687.133744,
            792.355984,
            841.339329,
            848.200523,
            694.178807,
            546.315579,
            817.995199,
            497.044582,
            657.5198905,
        ],
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'),
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
            loss_weight=2.0, type='CrossEntropyLoss', use_sigmoid=False),
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
sar_avg_mean = -639.664778
sar_avg_std = 657.5198905
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='val/multisen',
            img_path2='val/SAR_AVG_TIF',
            seg_map_path='val/label'),
        data_root='/home/hjkim/seg-challenge/MMSeg-YREB_last',
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
        data_root='/home/hjkim/seg-challenge/MMSeg-YREB_last',
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
            img_path='val/multisen',
            img_path2='val/SAR_AVG_TIF',
            seg_map_path='val/label'),
        data_root='/home/hjkim/seg-challenge/MMSeg-YREB_last',
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
vh_mean = -451.801359
vh_std = 497.044582
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
vv_mean = -827.528197
vv_std = 817.995199
work_dir = 'workdir/whisper/Gcbamr50_conv_normalize_256x256_upernet_last_v3'
