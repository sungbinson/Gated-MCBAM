import torch
from typing import Optional, Union
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList, dataset_aliases, get_classes, get_palette
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis.utils import ImageType
from mmseg.models import BaseSegmentor
from mmengine.model import BaseModel
from mmseg.apis import init_model, inference_model, show_result_pyplot
from collections import defaultdict
from typing import Sequence, Union
import tifffile as TIF
import numpy as np
from mmengine.dataset import Compose
from mmengine.model import BaseModel

import torch.nn.functional as F
import os
import pdb
from mmengine import Config
from PIL import Image


"""
1) upernet-r50-convnext-cbam-weighted-12Channel
2) upernet-r50-swin-cbam_weighted-12Channel
3) upernet-r50-convnext-cbam-weighted-6Channel
4) upernet-r50-convnext-cbam-weighted-9Channel
5) upernet-r50-convnext-gate_cbam-12Channel
6) upernet-r50-swin-gate_cbam-10Channel
7) upernet-r50-convnext-gate_cbam_weighted-12Channel
--------------------------------------------------------
1) 1 + 2 --> 67.25
2) 1 + 2 + 3 --> 67.59
3) 1 + 2 + 3 + 4 --> 67.31
4) 5 + 6 --> 66.88
5) 5 + 6 + 7 --> 67.22
6) 2 + 5 + 6 + 7 --> 67.26
--------------------------------------------------------
1) (1,2,3,5)
2) (1,2,3,6)
3) (1,2,3,7)

"""
# best_conv.pth, best.pth 다운 완료.
# /home/hjkim/seg-challenge/mmsegmentation/workdir/whisper/Gcbamr50_conv_normalize_256x256_upernet_last_v3/iter_12500.pth << pth만 바꾸면 바로 가능.
# ensemble 저장 폴더는 workdir/whisper/output

def multiple_inference_model(model: BaseSegmentor, img: ImageType):
    # prepare data
    data, is_batch = _preprare_multiple_data(img, model)
    
    # forward the model
    with torch.no_grad():
        results = model.test_step(data)

    return results if is_batch else results[0]

def _preprare_multiple_data(imgs: ImageType, model: BaseModel):
    
    cfg = model.cfg
    for t in cfg.test_pipeline:
        if t.get('type') == 'LoadAnnotations':
            cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)

    #for img in imgs:
    data_ = dict(img_path=imgs[0], img_path2=imgs[1])
    data_ = pipeline(data_)

    data['inputs'].append(data_['inputs'])
    data['data_samples'].append(data_['data_samples'])

    return data, is_batch

def _init_model(config, ckpt):
           
    model = init_model(config, ckpt, device='cuda:0')
    
    if not torch.cuda.is_available():
        model = revert_sync_batchnorm(model)    

    return model


if __name__ == "__main__":
    CONFIG = [
    '/home/hjkim/whispers2024_ouput/mmsegmentation/workdir/whisper/upernet-r50-convnext-cbam-weighted-12Channel/config.py',
    '/home/hjkim/whispers2024_ouput/mmsegmentation/workdir/whisper/upernet-r50-swin-cbam_weighted-12Channel/config.py',
    '/home/hjkim/whispers2024_ouput/mmsegmentation/workdir/whisper/Gcbamr50_conv_normalize_256x256_upernet_last_v3/config.py',          
    '/home/hjkim/whispers2024_ouput/mmsegmentation/workdir/whisper/Gcbamr50_swin_weight_256x256_upernet_last_v3/config.py'
    ]

    CKPT   = [
    '/home/hjkim/whispers2024_ouput/mmsegmentation/workdir/whisper/upernet-r50-convnext-cbam-weighted-12Channel/best.pth',
    '/home/hjkim/whispers2024_ouput/mmsegmentation/workdir/whisper/upernet-r50-swin-cbam_weighted-12Channel/best.pth',
    '/home/hjkim/whispers2024_ouput/mmsegmentation/workdir/whisper/Gcbamr50_conv_normalize_256x256_upernet_last_v3/best.pth',
    '/home/hjkim/whispers2024_ouput/mmsegmentation/workdir/whisper/Gcbamr50_swin_weight_256x256_upernet_last_v3/best.pth',          
    ]

    MSIPATH = '/home/hjkim/seg-challenge/MMSeg-YREB_last/test/MSI' 
    MSI10BPATH = '/home/hjkim/seg-challenge/MMSeg-YREB_last/test/multisen'
    SARPATH = '/home/hjkim/seg-challenge/MMSeg-YREB_last/test/SAR_AVG_TIF'

    SAVEPATH = ['/home/hjkim/whispers2024_ouput/mmsegmentation/workdir/whisper/output']
    for p in SAVEPATH:
        if not os.path.exists(p):
            os.mkdir(p)


    MSI12BSAR = sorted([[os.path.join(MSIPATH,    msi), os.path.join(SARPATH, sar)] for msi, sar in zip(os.listdir(MSIPATH),    os.listdir(SARPATH))])
    MSI10BSAR = sorted([[os.path.join(MSI10BPATH, msi), os.path.join(SARPATH, sar)] for msi, sar in zip(os.listdir(MSI10BPATH), os.listdir(SARPATH))])

   
    first  = _init_model(CONFIG[0], CKPT[0])
    second = _init_model(CONFIG[1], CKPT[1])
    third  = _init_model(CONFIG[2], CKPT[2])
    fourth = _init_model(CONFIG[3], CKPT[3])

    for idx, (msi12bsar, msi10bsar) in enumerate(zip(MSI12BSAR, MSI10BSAR)):
        first_result = multiple_inference_model(first, msi12bsar)[0]
        second_result = multiple_inference_model(second, msi12bsar)[0]
        third_result = multiple_inference_model(third, msi10bsar)[0]
        fourth_result = multiple_inference_model(fourth, msi10bsar)[0]
        first_logits = first_result.seg_logits.data
        second_logits = second_result.seg_logits.data
        third_logits = third_result.seg_logits.data
        fourth_logits = fourth_result.seg_logits.data

        # masks = (first_logits + second_logits + third_logits) / 3
        masks = (first_logits + second_logits + third_logits+ fourth_logits) / 4
        masks = masks.argmax(dim=0)
        masks = masks + 1
        savepath = os.path.join(SAVEPATH[0], msi10bsar[0].split('/')[-1])
        masks = masks.cpu().numpy()
        image = Image.fromarray(masks.astype(np.uint8))
        image.save(savepath)
        #print(msi10bsar[0].split('/')[-1])
        if idx % 100 == 0:
            print(f'{idx} is done....')

