# File: main.py
# Author: Ronil Pancholia
# Date: 4/6/19
# Time: 5:44 AM

import matplotlib
import numpy as np
import torch
from torch import nn, utils
from torch.utils.data import DataLoader
from torchvision import transforms

import dataset_utils
from face_dataset import FaceDataset

matplotlib.use('agg')
import matplotlib.pyplot as plt

from training_functions import get_model, fit
from config import RANDOM_SEED, BATCH_SIZE, NUM_WORKERS, TRAIN_EPOCHS, device

torch.manual_seed(RANDOM_SEED)


def save_plots(train_loss, train_acc, val_loss, val_acc):
    lst_iter = np.arange(1, TRAIN_EPOCHS + 1)
    plt.plot(lst_iter, train_loss, '-b', label='training loss')
    plt.plot(lst_iter, val_loss, '-r', label='validation loss')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc='upper right')
    plt.savefig("loss.png")
    plt.clf()

    plt.plot(lst_iter, train_acc, '-b', label='training accuracy')
    plt.plot(lst_iter, val_acc, '-r', label='validation accuracy')
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc='lower right')
    plt.savefig("accuracy.png")


# Normalize using mean and stddev of training set
transform_ops = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

dataset_loaders, dataset_sizes = dataset_utils.load_datasets(FaceDataset, transforms=transform_ops)
test_dataloaders, _ = dataset_utils.load_testset(FaceDataset, transforms=transform_ops)
test_dataloader = utils.data.DataLoader(test_dataloaders['test'], batch_size=BATCH_SIZE * 4, num_workers=NUM_WORKERS)

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
    for image, label in test_dataloaders['test']:
        image, label = image.to(device), label.to(device)
        outputs = model(image)
        preds = torch.argmax(outputs.data, 1)
        corrects += torch.sum(preds == label).item()
        preds = preds.cpu()
        results.extend(np.asarray(preds))

    print("Testing accuracy :", 100 * corrects / len(test_dataloaders['test'].dataset))

test_model()