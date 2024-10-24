import os
import numpy as np
import tifffile as tiff
import imageio
import mmcv

import pdb
def min_max_normalize(image, percentile=2):
    image = image.astype('float32')

    percent_min = np.percentile(image, percentile, axis=(0, 1))
    percent_max = np.percentile(image, 100-percentile, axis=(0, 1))

    _mask = np.mean(image, axis=2) != 0
    if image.shape[1] * image.shape[0] - np.sum(_mask) > 0:
        mdata = np.ma.masked_equal(image, 0, copy=False)
        mdata = np.ma.filled(mdata, np.nan)
        percent_min = np.nanpercentile(mdata, percentile, axis=(0, 1))

    norm = (image - percent_min) / (percent_max - percent_min)
    norm[norm < 0] = 0
    norm[norm > 1] = 1
    norm = (norm * 255).astype('uint8') * _mask[:, :, np.newaxis]
    return norm



base_dir = "/home/min_jeongho/dataset/sentienl-2"
target_dir = "/home/min_jeongho/mmsegmentation/data/sen2-RGB/"


colors_bgr = ["B02.tif", "B03.tif", "B04.tif"]
# s1_dir = os.path.join(base_dir, "JRCPerm")
# target_dir = os.path.join(target_dir,"JRCPerm")    

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
for dir_path, dirnames,filenames in os.walk(base_dir):
    blue = None
    green = None
    red = None
    for filename in filenames:
        file_path = os.path.join(dir_path, filename)

        
        # Blue
        if filename.endswith(colors_bgr[0]):
            blue = tiff.imread(file_path)
            
        # Green
        elif filename.endswith(colors_bgr[1]):
            green = tiff.imread(file_path)
            
        # Red
        elif filename.endswith(colors_bgr[2]):
            red = tiff.imread(file_path)
    
    if blue is not None and green is not None and red is not None:
        new_image = np.stack((red, green, blue), axis=-1)
        
        new_image = min_max_normalize(new_image)
        
        
        new_filename = os.path.basename(dir_path) + '.png'
        
        new_file_path = os.path.join(target_dir, new_filename)
                
        imageio.imwrite(new_file_path, new_image.astype(np.uint8))
    


print("Done")