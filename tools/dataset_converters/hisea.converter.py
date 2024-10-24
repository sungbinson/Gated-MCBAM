import os
from PIL import Image
import numpy as np
import mmcv

base_dir = "/home/min_jeongho/mmsegmentation/data/HISEA1"  


folders = ['train', 'test', 'val']


for folder in folders:
    label_dir = os.path.join(base_dir, folder, 'label_1D')
    
    
    for filename in os.listdir(label_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):  
            file_path = os.path.join(label_dir, filename)
            
            
            label_img = mmcv.imread(file_path)
            # label_np = np.where(label_np == 255, 1, label_np)
            
            
            # label_img = Image.fromarray(label_np)
            mmcv.imwrite(label_img[:,:,0] // 255, file_path)

print("Done")