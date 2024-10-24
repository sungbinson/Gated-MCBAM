_base_ = ['/home/hjkim/seg-challenge/mmsegmentation/configs/mmsfe2former/cbamr50_swin_normalize_256x256_upernet_80k_jhmin_celoss.py']

conv_pretrained = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'  # noqa
load_from = "/home/hjkim/seg-challenge/mmsegmentation/workdir/whisper/Gcbamr50_conv_normalize_256x256_upernet_last_pretrained/10ch_best.pth"
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

    

    )