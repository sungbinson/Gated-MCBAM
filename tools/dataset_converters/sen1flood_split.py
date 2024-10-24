import os
import shutil
import pandas as pd


base_dir = "/home/min_jeongho/dataset/sen1floods11_512x512/sen1floods/"
csv_files = ["flood_train_data.csv", "flood_valid_data.csv", "flood_test_data.csv"]
folders = ["train", "val", "test"]

target_dir = "/home/min_jeongho/mmsegmentation/data/sen1floods11_512x512"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for csv_file, folder in zip(csv_files, folders):
    
    csv_path = os.path.join(base_dir, csv_file)
    
    
    with open(csv_path, 'r') as f:
        filenames = f.read().splitlines()
    

    # folder_path = os.path.join(base_dir, folder)
    target_path = os.path.join(target_dir, folder)
    label_dir = os.path.join(target_path, "Labels")
    s1_dir = os.path.join(target_path, "S1")
    # print(s1_dir)
    
    
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(s1_dir, exist_ok=True)

    for filename in filenames:
        f_name = filename.split(",")[0]
        target_filename = f_name.replace("_S1Hand.tif", ".tif")

        label_file = os.path.join(filename.split(",")[1])
        s1_file = os.path.join(filename.split(",")[0])
        # print(filename, target_filename)
        # exit()
        # if os.path.exists(label_file):
        shutil.copy(os.path.join(base_dir,"Labels",label_file), os.path.join(label_dir, target_filename))
        # # if os.path.exists(s1_file):
        shutil.copy(os.path.join(base_dir,"S1",s1_file), os.path.join(s1_dir, target_filename))
        


print("Done")