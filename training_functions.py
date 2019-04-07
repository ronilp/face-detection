# File: training_functions.py
# Author: Ronil Pancholia
# Date: 4/6/19
# Time: 2:07 AM

import sys
import torch
from torch import optim
from tqdm import tqdm

from config import device, BASE_LR, REGULARIZATION_LAMBDA
from model import LeNet5


def loss_batch(model, criterion, x, y, opt=None):
    outputs = model(x)
    loss = criterion(outputs, y)

    lmda = torch.tensor(REGULARIZATION_LAMBDA)
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l2_reg += torch.norm(param)

    loss += lmda * l2_reg

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    preds = torch.argmax(outputs.data, 1)
    corrects = torch.sum(preds == y)
    return loss.item(), len(x), corrects


def fit(num_epochs, model, criterion, opt, train_dataloader, val_dataloader=None):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        print("\nEpoch " + str(epoch + 1))

        running_loss = 0.0
        model.train()
        running_corrects = 0
        for image, label in train_dataloader:
            image, label = image.to(device), label.to(device)
            losses, nums, corrects = loss_batch(model, criterion, image, label, opt)
            running_loss += losses
            running_corrects += corrects

        train_loss.append(running_loss / len(train_dataloader.dataset))
        train_acc.append(running_corrects.item() / (len(train_dataloader.dataset)))
        print("Training loss:", train_loss[-1], "| Training accuracy: %.2f" % train_acc[-1])

        if val_dataloader == None:
            continue

        model.eval()
        running_corrects = 0
        with torch.no_grad():
            for image, label in val_dataloader:
                image, label = image.to(device), label.to(device)
                losses, nums, corrects = loss_batch(model, criterion, image, label)
                running_loss += losses
                running_corrects += corrects

        val_loss.append(running_loss / len(val_dataloader.dataset))
        val_acc.append(running_corrects.item() / (len(val_dataloader.dataset)))
        print("Validation loss:", val_loss[-1], " | Validation accuracy: %.2f" % val_acc[-1])

    return train_loss, train_acc, val_loss, val_acc

def get_model():
    model = LeNet5()
    model.to(device)
    return model, optim.SGD(model.parameters(), lr=BASE_LR)