import os
import rasterio
import numpy as np
from tqdm import tqdm

def extract_rgb_nir(input_path, output_path):
   with rasterio.open(input_path) as src:
       
       # Select specific band indices (1-based indexing in rasterio)
       band_indices = [2, 3, 4, 5, 6, 7, 9, 8, 11, 12]
       
       # Read only required bands 
       data = src.read(band_indices)
       
       # Update profile settings
       profile = src.profile
       profile.update(count=10, dtype=rasterio.float32)
       
       # Write to new file
       with rasterio.open(output_path, 'w', **profile) as dst:
           dst.write(data.astype(rasterio.float32))
           
   # Verify the output file has correct number of channels
   with rasterio.open(output_path) as check:
       if check.count != 10:
           print(f"Warning: {output_path} does not have 4 channels. It has {check.count} channels.")
           exit()

def process_dataset(input_folder, output_folder):
   # Create output folder if it doesn't exist
   if not os.path.exists(output_folder):
       os.makedirs(output_folder)
   
   # Process all tif files in input folder
   for filename in tqdm(os.listdir(input_folder)):
       if filename.endswith('.tif'):
           input_path = os.path.join(input_folder, filename)
           output_path = os.path.join(output_folder, filename)
           extract_rgb_nir(input_path, output_path)

# Main execution section
def main():
   base_folder = '/home/hjkim/seg-challenge/MMSeg-YREB'
   
   for dataset in ['train', 'val', 'test']:
       input_folder = os.path.join(base_folder, dataset, 'MSI')
       output_folder = os.path.join(base_folder, dataset, 'multisen')
       
       print(f"Processing {dataset} dataset...")
       process_dataset(input_folder, output_folder)
       print(f"Finished processing {dataset} dat