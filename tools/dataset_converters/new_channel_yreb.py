import os
import numpy as np
import tifffile as tiff
import imageio
import mmcv
import scipy.ndimage as ndimage
folders = ["test"]
def replace_nan_with_zero(mask):

    mask[np.isnan(mask)] = 0
    return mask
def replace_value(image, old_value, new_value):

    image[image == old_value] = new_value
    return image
def min_max_normalize(image, percentile=2):
    # Input: [H, W, C]
        # tif 값의 nodata != 0 인경우
    # _mask = np.mean.. line의 !=0 값을 바꿀것.
        # nodata = rasterio.open(filename).meta['nodata']
    image = image.astype('float32')

    percent_min = np.percentile(image, percentile, axis=(0, 1))
    percent_max = np.percentile(image, 100-percentile, axis=(0, 1))

    _mask = np.mean(image, axis=2) != 0
    if image.shape[1] * image.shape[0] - np.sum(_mask) > 0:
        mdata = np.ma.masked_equal(image, 0, copy=False)
        mdata = np.ma.filled(mdata, np.nan)
        percent_min = np.nanpercentile(mdata, percentile, axis=(0, 1))

    norm = (image-percent_min) / (percent_max - percent_min)
    norm[norm < 0] = 0
    norm[norm > 1] = 1
    norm = (norm * 255).astype('uint8') * _mask[:, :, np.newaxis]
    return norm

def average_channel_sar(vv, vh):
    average = (vv + vh) / 2
    return np.stack([vv, vh, average], axis=-1)

def difference_channel_sar(vv, vh):
    difference = np.abs(vv - vh)
    return np.stack([vv, vh, difference], axis=-1)

def replace_nan_with_mean(image):
   
    nan_mask = np.isnan(image)
    temp_image = np.where(nan_mask, 0, image)
    mean_filtered = ndimage.generic_filter(temp_image, np.nanmean, size=3)
    image[nan_mask] = mean_filtered[nan_mask]
    
    return image

import pdb
for folder in folders:
    base_dir = "/home/hjkim/seg-challenge/MMSeg-YREB"
    target_dir = "/home/hjkim/seg-challenge/MMSeg-YREB_v5"
    s1_dir = os.path.join(base_dir, folder, "SAR")
    target_dir = os.path.join(target_dir, folder, "SAR_AVG_TIF")



    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename in os.listdir(s1_dir):
        if filename.endswith('.tif'):
            file_path = os.path.join(s1_dir, filename)
            
            image = tiff.imread(file_path)
    
            vv = image[:,:,0]
            vh = image[:,:,1]

            # new_image = replace_nan_with_zero(image)
            # new_image = replace_value(new_image, old_value=-1, new_value=0)
            # nan -> mean
            # vv = min_max_normalize(vv).squeeze()
            # vh = min_max_normalize(vh).squeeze()
            # vv = replace_nan_with_mean(vv)
            # vh = replace_nan_with_mean(vh)
            # NewBand1 = (VH - VV) / (VH + VV)
            # new_band1 = np.abs((vh - vv)) / (vh + vv + 1e-6)  
            new_band2 = (vv+vh) / 2
            # NewBand2 = sqrt((VH^2 + VV^2) / 2)

            # new_band2 = np.sqrt((vh ** 2 + vv ** 2) / 2)
            
            new_image = np.stack((vv, vh, new_band2), axis=-1)
            # new_image = average_channel_sar(vv, vh)
            # new_image = difference_channel_sar(vv,vh)
            # new_image = np.stack((new_band1, new_band1, new_band1), axis=-1)
           
            new_filename = filename.replace('.tif', '.tif')
            new_file_path = os.path.join(target_dir , new_filename)
            print(new_file_path)
            
            # print(file_path)
            # print(new_file_path)
            # exit()
            
            # imageio.imwrite(new_file_path, new_image.astype(np.uint8))
            tiff.imwrite(new_file_path, new_image)

print("Done")