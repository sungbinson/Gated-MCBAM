import os
from PIL import Image
import numpy as np
import mmcv
from scipy.ndimage import generic_filter
from scipy.special import gamma
from scipy.ndimage import median_filter
import cv2

from scipy.ndimage import uniform_filter, variance
from skimage import io, img_as_float
base_dir = "/home/min_jeongho/mmsegmentation/data/HISEA1" 
target_dir = "/home/min_jeongho/mmsegmentation/data/HISEA1_despeckled" 

def lee_filter(img, size):
    
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = np.var(img)
    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    
    return img_output

def frost_filter(img, size, damping_factor=2.0):
    def filter_function(values):
        center = values[len(values) // 2]
        weights = np.exp(-damping_factor * np.abs(values - center) / np.mean(values))
        return np.sum(weights * values) / np.sum(weights)

    return generic_filter(img, filter_function, size=size)


def gamma_map_filter(img, size, enl=1.0):
    def filter_function(values):
        mean = np.mean(values)
        var = np.var(values)
        b = mean / (var / mean - 1)
        a = mean / b
        return a / (1 + 1 / enl)

    return generic_filter(img, filter_function, size=size)

def median(sar_image):
    median_filtered_image = median_filter(sar_image, size=5)
    return median_filtered_image 

if not os.path.exists(target_dir):
    os.makedirs(target_dir)


folders = ['train', 'test', 'val']


for folder in folders:
    label_dir = os.path.join(base_dir, folder, 'image')
    target = os.path.join(target_dir, folder, 'image')
    
    if not os.path.exists(target):
        os.makedirs(target)

    for filename in os.listdir(label_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):  
            file_path = os.path.join(label_dir, filename)
            target_file_path = os.path.join(target, filename) 
            
            # label_img = mmcv.imread(file_path)
            sar_image = img_as_float(Image.open(file_path))
            denoised_image = lee_filter(sar_image, size=5)
            
            normalized_image = cv2.normalize(denoised_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            
            unsharp_image_8bit = (255 * normalized_image).astype(np.uint8)
            unsharp_image_pil = Image.fromarray(unsharp_image_8bit)
            # label_np = np.where(label_np == 255, 1, label_np)
            
            # label_img = Image.fromarray(label_np)
            unsharp_image_pil.save(target_file_path)

print("Done")