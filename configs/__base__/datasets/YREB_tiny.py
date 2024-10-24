dataset_type = 'YREBtinyDataset'
data_root = '/home/mjh/Dataset/MMSeg-YREB-tiny'
'''
tiny
RGB_NIR Normalization Values:
R - Mean: 1351.283074, Std: 456.481785
G - Mean: 1140.649691, Std: 506.950743
B - Mean: 1046.667887, Std: 520.133303
NIR - Mean: 2107.813150, Std: 613.706617

SAR Normalization Values:
VV - Mean: -1473.350952, Std: 429.976438
VH - Mean: -793.996916, Std: 364.819485
'''

vv_mean , vh_mean = -1473.350952 , -793.996916
vv_std , vh_std = 429.976438, 364.819485
# SAR AVG
sar_avg_mean = (vv_mean + vh_mean) / 2
sar_avg_std = 281.589694
crop_size = (128, 128)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[1046.667887, 1140.649691, 1351.283074, -1473.350952, -793.996916, (vv_mean + vh_mean) / 2],
    std=[520.133303, 506.950743, 456.481785, 429.976438, 364.819485, 281.589694],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))
    
# dataset config
train_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    # dict(type='DebugTransform'),
    dict(type='LoadAnnotations'),
    dict(
        type='MultiImgRandomResize',
        scale=(256, 256),  
        ratio_range=(0.5, 2)),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5),
    # dict(type='MultiImgPhotoMetricDistortion'),
    # dict(type='MultiImgNormalize', **img_norm_cfg),
    # dict(type='MultiImgPad', size=crop_size),
    dict(type='ConcatMultiInput'),
    dict(type='PackMultiInputs') ]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_pipeline=[
            dict(type='LoadMultipleRSImageFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='SARChannelAugmentation', channel ='AVG'),
            dict(type='ConcatMultiInput'),
            dict(type='PackMultiInputs')
        ]
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_pipeline=[
            dict(type='LoadMultipleRSImageFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='SARChannelAugmentation'),
            dict(type='ConcatMultiInput'),
            dict(type='PackMultiInputs')
        ]
test_dataloader =dict(dataset=dict(pipeline=test_pipeline))
test_cfg = dict(type='TestLoop')

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]


train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/RGB_NIR', img_path2='train/SAR', seg_map_path='train/label'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/RGB_NIR', img_path2='val/SAR',
            seg_map_path='val/label'),
        pipeline=test_pipeline))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_dataloader = val_dataloader
test_evaluator = val_evaluator