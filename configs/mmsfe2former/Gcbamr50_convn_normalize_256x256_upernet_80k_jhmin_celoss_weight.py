_base_ = ['/home/hjkim/seg-challenge/mmsegmentation/configs/mmsfe2former/cbamr50_swin_normalize_256x256_upernet_80k_jhmin_celoss.py']

conv_pretrained = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'  # noqa
find_unused_parameters=True
model = dict(
    backbone=dict(
        _delete_ = True,
        type='MMEncoder_v2',
        convn_arch='base',
        convn_out_indices=[0, 1, 2, 3],
        convn_drop_path_rate=0.4,
        convn_layer_scale_init_value=1.0,
        convn_gap_before_final_norm=False,
        convn_init_cfg=dict(
                    type='Pretrained', checkpoint=conv_pretrained, prefix='backbone.')),

    
    decode_head=dict(
         loss_decode=dict(
             class_weight=[1.0, 5.0, 3.0, 2.0, 10.0, 4.0, 4.0, 4.0, 7.0],
            # class_weight=[1.0, 8.0, 5.0, 2.0, 20.0, 7.0, 7.0, 7.0, 15.0],
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0) ),

    auxiliary_head=dict(
         loss_decode=dict(
            class_weight=[1.0, 5.0, 3.0, 2.0, 10.0, 4.0, 4.0, 4.0, 7.0],
            # class_weight=[1.0, 8.0, 5.0, 2.0, 20.0, 7.0, 7.0, 7.0, 15.0],
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    )    
    )