# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import HMDB51
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import av
import time
import os
from tqdm import tqdm
from frame_dataset import FrameDataset
import copy
import json

plt.ion()  # interactive mode

hmdb_data_dir = "/home/jrola/PycharmProjects/pytorch_CTM/hmdb4_org"
hmdb_label_dir = "/home/jrola/PycharmProjects/pytorch_CTM/testTrainMulti_7030_splits"
hmdb_frames_dir = "/home/jrola/PycharmProjects/pytorch_CTM/hmdb4_frames"
train_labels = "/home/jrola/PycharmProjects/pytorch_CTM/hmdb4_labels80.csv"
test_labels = "/home/jrola/PycharmProjects/pytorch_CTM/hmdb4_labels20.csv"

batch_size = 32

# create train loader (allowing batches and other extras)
# train_dataset = HMDB51(hmdb_data_dir, hmdb_label_dir, frames_per_clip=frames_per_clip, step_between_clips=step_between_clips, train=True, transform=tfs)
train_dataset = FrameDataset(
    csv_file=train_labels,
    root_dir=hmdb_frames_dir,
)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# create test loader (allowing batches and other extras)
# test_dataset =  HMDB51(hmdb_data_dir, hmdb_label_dir, frames_per_clip=frames_per_clip, step_between_clips=step_between_clips, train=False, transform=tfs)
test_dataset = FrameDataset(
    csv_file=test_labels,
    root_dir=hmdb_frames_dir,
)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""função de treino do modelo"""


# Train Network
def train_model(model, criterion, optimizer, scheduler, stats, num_epochs=4):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase == 'train':
                data, tag = next(iter(train_dataloader))
                for data, tag in tqdm(train_dataloader):
                    data = data.to(device)
                    tag = tag.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(data)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, tag)
                        # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * data.size(0)
                    running_corrects += torch.sum(preds == tag)

                    scheduler.step()
                    # unindent this!
                epoch_loss = running_loss / len(train_dataset)
                epoch_acc = running_corrects.double() / len(train_dataset)
                stats[phase + '_loss'].append(float(epoch_loss))
                stats[phase + '_acc'].append(float(epoch_acc))
            else:
                for data, tag in tqdm(test_dataloader):
                    data, tag = next(iter(test_dataloader))
                    data = data.to(device)
                    tag = tag.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(data)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, tag)

                    # statistics
                    running_loss += loss.item() * data.size(0)
                    running_corrects += torch.sum(preds == tag)
                    # unindent this!
                epoch_loss = running_loss / len(test_dataloader)
                epoch_acc = running_corrects.double() / len(test_dataset)
                stats[phase + '_loss'].append(float(epoch_loss))
                stats[phase + '_acc'].append(float(epoch_acc))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


## prepare the model


class_name_to_label_path = '/home/jrola/PycharmProjects/pytorch_CTM/class_name_to_label_4.json'
f = open(class_name_to_label_path)
classes = json.load(f)
f.close()

model_ft = torchvision.models.video.r3d_18(pretrained=True, progress=True)

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(512, len(classes))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                amsgrad=False)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.1)

## train and evaluate

stats = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
num_epochs = 4
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, stats, num_epochs)

## Show results graphs

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(0, 10, 0.1)
y = stats['train_loss']
z = stats['val_loss']
ax.plot(y, color='blue')
ax.plot(z, color='grey')
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
y = stats['train_acc']
z = stats['val_acc']
ax.plot(y, color='blue')
ax.plot(z, color='grey')
plt.show()
