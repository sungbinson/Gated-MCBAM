_base_ = ['./cbamr50_swin_normalize_256x256_upernet_80k.py']

data_root  = '/home/hjkim/seg-challenge/MMSeg-YREB'
crop_size = (256, 256)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(
            img_path='train/MSI',
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
            dict(alpha=0.7, prob=0.5, type='MultiImgCutOut'),
            dict(type='ConcatMultiInput'),
            dict(type='PackMultiInputs'),
        ],
        type='YREBtinyDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))

optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=3e-4,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9),
    type='OptimWrapper')