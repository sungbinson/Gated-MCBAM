_base_ = ['/home/hjkim/seg-challenge/mmsegmentation/configs/mmsfe2former/cbamr50_swin_normalize_256x256_upernet_80k_jhmin_celoss.py']

load_from = "/home/hjkim/seg-challenge/mmsegmentation/workdir/whisper/swin_best/swin_best.pth"
find_unused_parameters=True
model = dict(
    backbone=dict(
        type='MMEncoder_v5_swinGCBAM',
        swin_in_channel = 10,
        ),
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