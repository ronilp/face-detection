# File: face_dataset.py
# Author: Ronil Pancholia
# Date: 4/6/19
# Time: 3:23 PM
import os
import cv2
import torch
from torch.utils.data.dataset import Dataset

import config
import dataset_utils


class FaceDataset(Dataset):
    def __init__(self, dir_path, mode, transforms=None):
        # Set transforms
        self.transforms = transforms
        print("Loading dataset from ", dir_path)

        self.img_arr = []
        self.label_arr = []
        self.class_counts = {}
        self.classes, self.class_to_idx = dataset_utils.find_classes(dir_path)
        for class_name in config.ALLOWED_CLASSES:
            self.class_counts[class_name] = 0

        for class_name in config.ALLOWED_CLASSES:
            class_path = os.path.join(dir_path, class_name)
            for file_name in os.listdir(class_path):
                if not file_name.endswith(".jpg"):
                    continue

                self.img_arr.append(os.path.join(class_path, file_name))
                self.label_arr.append(self.class_to_idx[class_name])
                self.class_counts[class_name] += 1

        # Calculate len
        self.data_len = len(self.img_arr)

        print(mode + " class counts : " + str(self.class_counts))

    def __getitem__(self, index):
        input = cv2.imread(self.img_arr[index])
        input = cv2.resize(input, (60, 60))

        if self.transforms is not None:
            input = self.transforms(input)

        # Transform image to tensor
        input_tensor = torch.Tensor(input)

        # Get label of the image
        label_tensor = torch.tensor(self.label_arr[index])

        return (input_tensor, label_tensor)

    def __len__(self):
        return self.data_len