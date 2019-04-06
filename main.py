# File: main.py
# Author: Ronil Pancholia
# Date: 4/6/19
# Time: 5:44 AM

import matplotlib
import numpy as np
import torch
import torchvision
from torch import nn, utils
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms

import dataset_utils
from face_dataset import FaceDataset

matplotlib.use('agg')
import matplotlib.pyplot as plt

from training_functions import get_model, fit
from config import RANDOM_SEED, dataset_size, train_ratio, BATCH_SIZE, num_workers, TRAIN_EPOCHS, device

torch.manual_seed(RANDOM_SEED)


def save_plots(train_loss, train_acc, val_loss, val_acc):
    lst_iter = np.arange(1, TRAIN_EPOCHS + 1)
    plt.plot(lst_iter, train_loss, '-b', label='training loss')
    plt.plot(lst_iter, val_loss, '-r', label='validation loss')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig("loss.png")

    plt.plot(lst_iter, train_acc, '-b', label='training accuracy')
    plt.plot(lst_iter, val_acc, '-r', label='validation accuracy')
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc='bottom right')
    plt.show()
    plt.savefig("accuracy.png")


train_size = int(dataset_size * train_ratio)
val_size = dataset_size - train_size

# Normalize using mean and stddev of training set
transform_ops = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

dataset_loaders, dataset_sizes = dataset_utils.load_datasets(FaceDataset)

test_dataloaders, _ = dataset_utils.load_testset(FaceDataset)
test_dataloader = utils.data.DataLoader(test_dataloaders['test'], batch_size=BATCH_SIZE * 4, num_workers=num_workers)

criterion = nn.CrossEntropyLoss()

def save_results(train_loss, train_acc, val_loss, val_acc):
    f = open("results.txt", "w")
    f.write(str(train_loss[-1]) + "\n")
    f.write(str(val_loss[-1]) + "\n")
    f.write(str(train_acc[-1]) + "\n")
    f.write(str(val_acc[-1]) + "\n")
    f.close()

# Train model
model, opt = get_model()
train_loss, train_acc, val_loss, val_acc = fit(TRAIN_EPOCHS, model, criterion, opt, dataset_loaders['train'], dataset_loaders['val'])

save_plots(train_loss, train_acc, val_loss, val_acc)
save_results(train_loss, train_acc, val_loss, val_acc)

# Test model
def test_model():
    model.eval()
    results = []
    corrects = 0
    for image, label in test_dataloader:
        image, label = image.to(device), label.to(device)
        outputs = model(image)
        preds = torch.argmax(outputs.data, 1)
        corrects += torch.sum(preds == label).item()
        preds = preds.cpu()
        results.extend(np.asarray(preds))

    print("Testing accuracy :", 100 * corrects / len(test_dataloader.dataset))

    def convert_to_one_hot(arr):
        b = np.zeros((10000, 10))
        b[np.arange(10000), arr] = 1
        return b.astype(int)

    # Save results
    results = convert_to_one_hot(results)
    np.savetxt("mnist.csv", results, delimiter=",", fmt="%i")
