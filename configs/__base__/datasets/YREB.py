dataset_type = 'YREBDataset'
data_root = '/home/whisper2024/MMSeg-YREB_v3'

crop_size = (256,256)
train_pipeline = [
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
        cat_max_ratio=0.75, crop_size=crop_size, type='MultiImgRandomCrop'),
    dict(prob=0.5, type='MultiImgRandomFlip'),
    dict(type='ConcatMultiInput'),
    dict(type='PackMultiInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/RGB_NIR', img_path2='train/SAR_AVG_TIF', seg_map_path='train/label'),
        pipeline=train_pipeline)
)
tta_model = dict(type='SegTTAModel')
val_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(keep_ratio=True, scale=(
        256,
        256,
    ), type='Resize'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(type='ConcatMultiInput'),
    dict(type='PackMultiInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='val/RGB_NIR',
            img_path2='val/SAR_AVG_TIF',
            seg_map_path='val/label'),
        data_root=data_root,
        pipeline=val_pipeline,
        type=dataset_type),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mFscore'])
test_dataloader = val_dataloader
test_evaluator = val_evaluator

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