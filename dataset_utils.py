# File: dataset_utils.py
# Author: Ronil Pancholia
# Date: 4/6/19
# Time: 2:03 AM

import os
import torch
import multiprocessing

import config


def load_datasets(DatasetClass, transforms=None, datasets=None):
    if datasets is None:
        datasets = {x: DatasetClass(os.path.join(config.DATA_DIR, x), x, transforms) for x in ['train', 'val']}
    dataset_loaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=config.BATCH_SIZE, shuffle=True,
                                       num_workers=multiprocessing.cpu_count()) for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    return dataset_loaders, dataset_sizes


def load_testset(DatasetClass, transforms=None, datasets=None):
    if datasets is None:
        datasets = {x: DatasetClass(os.path.join(config.DATA_DIR, x), x, transforms) for x in ['test']}
    dataset_loaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=config.BATCH_SIZE * 4, shuffle=False,
                                       num_workers=multiprocessing.cpu_count()) for x in ['test']}
    dataset_sizes = {x: len(datasets[x]) for x in ['test']}
    return dataset_loaders, dataset_sizes


def load_datasets_from_csv(DatasetClass, transforms=None):
    datasets = {x: DatasetClass(config.DATA_DIR, x, transforms) for x in ['train', 'val']}
    return load_datasets(DatasetClass, transforms, datasets)


def load_testset_from_csv(DatasetClass, transforms=None):
    datasets = {x: DatasetClass(config.DATA_DIR, x, transforms) for x in ['test']}
    return load_testset(DatasetClass, transforms, datasets)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if
               os.path.isdir(os.path.join(dir, d)) and d in config.ALLOWED_CLASSES]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx