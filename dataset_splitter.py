# File: dataset_splitter.py
# Author: Ronil Pancholia
# Date: 4/6/19
# Time: 4:45 PM

import os
from shutil import copyfile
import random
dir_path = "data/processed"
output_path = "data"

def split_dir(train_split, val_split, class_name):
    orig_path = os.path.join(dir_path, class_name)
    file_list = os.listdir(orig_path)
    train_len = int(len(file_list) * train_split)
    val_len = int(len(file_list) * val_split)

    random.shuffle(file_list)
    train = file_list[0:train_len]
    val = file_list[train_len:train_len+val_len]
    test = file_list[train_len+val_len:]

    print("train: ", len(train))
    print("val: ", len(val))
    print("test: ", len(test))

    copy_files(train, "train/", class_name)
    copy_files(val, "val/", class_name)
    copy_files(test, "test/", class_name)

def copy_files(file_list, mode, class_name):
    path = os.path.join(dir_path, class_name)
    for file_name in file_list:
        dest = os.path.join(output_path, mode + class_name)
        print("Copying :", os.path.join(path, file_name), " to ", os.path.join(dest, file_name))
        copyfile(os.path.join(path, file_name), os.path.join(dest, file_name))


split_dir(0.6, 0.2, 0.2, "face")
split_dir(0.6, 0.2, 0.2, "no_face")