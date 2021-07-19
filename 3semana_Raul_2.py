#!/usr/bin/env python
# coding: utf-8

# function ClickConnect(){
#     console.log("Clicked on connect button"); 
#     document.querySelector("colab-connect-button").click()
# }
# setInterval(ClickConnect,60000)

# ### Import packages

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().system(' cp -vR /content/drive/MyDrive/mini_dataset /teste')


# In[3]:



import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import cv2
import os
from os import path
import json
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

import pandas as pd
from PIL import Image
import random
import torchvision.transforms as T
from torch.utils.data import Dataset

#if gpu, use it
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ### Extract frames from videos and place them inside video name folder

# In[4]:


def extract_frames(file_path, target_dir):
   if not os.path.exists(target_dir):
       os.makedirs(target_dir)

   vidcap = cv2.VideoCapture(file_path)
   success, image = vidcap.read()
   count = 0

   while success:
       frame_path = os.path.join(target_dir, f'{count}.jpg')
       cv2.imwrite(frame_path, image)     # save frame as JPEG file
       count += 1
       success, image = vidcap.read()
  

root = '/teste'
frame_root = '/teste/data/frames'
labels_path = '/teste/data/labels.csv'
class_name_to_label_path = '/teste/data/class_name_to_label.json'

#read files
files = []

for class_name in os.listdir(root):
   for video_name in os.listdir(os.path.join(root, class_name)):
        files.append([os.path.join(class_name, video_name), class_name])

#normalize labels
class_name_to_label = {}
current_label = -1

for vid in files:
   label = class_name_to_label.get(vid[1], -1)

   if label == -1:
       current_label += 1
       class_name_to_label[vid[1]] = current_label
       label = current_label

   vid[1] = label


# save file paths
if not os.path.exists(os.path.split(labels_path)[0]):
    os.makedirs(os.path.split(labels_path)[0])

f = open(labels_path, 'w')

f.write('path,label\n')

for vid in files:
   f.write(f'{vid[0]},{vid[1]}\n')

f.close()

# save label normalization
if not os.path.exists(os.path.split(class_name_to_label_path)[0]):
   os.makedirs(os.path.split(class_name_to_label_path)[0])

with open(class_name_to_label_path, 'w') as json_file:
   json.dump(class_name_to_label, json_file, indent=4)

# extract frames
for i, vid in enumerate(files):
   file_path = os.path.join(root, vid[0])
   target_dir = os.path.join(frame_root, vid[0])

   extract_frames(file_path, target_dir)

   print(f'{i+1}/{len(files)}')


# ### Split labels csv file into train and validate - 80/20

# In[5]:


import pandas as pd

def split(label, path):
   train = label.sample(frac=0.8, random_state=201)
   val = label.drop(train.index)
   train.to_csv(path + '/train.csv', mode='a', header=False)
   val.to_csv(path + '/val.csv', mode='a', header=False)


root = '/teste/data'
frame_root = '/teste/data/frames'
labels_path = '/teste/data/labels.csv'
class_name_to_label_path = '/teste/data/class_name_to_label.json'

labels_data = pd.read_csv(labels_path)
last_label = labels_data.tail(1)['label']
last_label = last_label.iloc[0]
max_labels = labels_data.nunique()
max_labels = max_labels['label']
print(max_labels)
index = 0

for i in range(max_labels):
   label = labels_data[labels_data['label'] == index]
   split(label, root)
   index +=1


# ### custom dataset

# In[6]:



train_csv_path = '/teste/data/train.csv'
validate_csv_path = '/teste/data/val.csv'
root_path = '/teste/data/frames'

class MyDataset(Dataset):
    def __init__(self, frames_csv_file, root_dir):
        self.frames_csv = pd.read_csv(frames_csv_file)
        self.root_dir = root_dir
        self.slice_size = 10

    def __len__(self):
        return len(self.frames_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
      
      #need to take 10 frames and stack them on 1 tensor                                                    
      #()-> second index has to be 1 because id column (3 columns) 
      #()-> why have to be NOT??? is wrong!! but works..
        all_frames = [f for f in os.listdir(os.path.join(self.root_dir, str(self.frames_csv.iloc[idx, 1]))) if not os.path.isfile(f)]
      #random get 10 consecutive frames from all video frames in folder
        start = random.randrange(len(all_frames) - self.slice_size)
        frames = all_frames[start: start + self.slice_size]

        images = []
      #load the images
        for frame in frames:
            path = os.path.join(self.root_dir, str(self.frames_csv.iloc[idx, 1])) + '/' + frame
            images.append(Image.open(path).convert('RGB'))
        transform = transforms.Compose([
                    T.Resize((240, 320),),
                    T.ToTensor(),
                    T.Normalize(mean = [0.43216, 0.394666, 0.37645], std = [0.22803, 0.22145, 0.216989])])
        #apply tansforms and condense all the images in 1 tensor
        tensors = []
        for image in images:
            tensors.append(transform(image))            
        final_image = torch.stack(tensors, dim=1)
        tag = int(self.frames_csv.iloc[idx, 2])
        return final_image, tag

train_dataset = MyDataset(train_csv_path, root_path)
validate_dataset = MyDataset(validate_csv_path, root_path)


# ## dataloaders

# In[7]:



from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=6, shuffle=True)


# ## training loop

# In[29]:


def train_loop(dataloader, model, loss_fn, optimizer, stats):
    size = len(dataloader.dataset)
    count = 0
    for X,y in dataloader:
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stats['train_loss'].append(loss.item())
        if count % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}")
        count += 1

def test_loop(dataloader, model, loss_fn, stats):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    stats['val_loss'].append(test_loss)
    stats['val_acc'].append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# ## prepare the model

# In[18]:


class_name_to_label_path = '/teste/data/class_name_to_label.json'
f = open(class_name_to_label_path)
classes = json.load(f)
f.close()

model_ft = torchvision.models.video.r3d_18(pretrained=True, progress=True)

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(512, len(classes))

model_ft = model_ft.to(device)

loss_fn = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)


# In[10]:


## train and evaluate


# In[30]:


stats = {'train_loss':[], 'val_loss':[], 'val_acc':[]}
num_epochs=3
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model_ft, loss_fn, optimizer_ft, stats)
    test_loop(validate_dataloader, model_ft, loss_fn, stats)
print("Done!")


# ## Show results graphs

# In[33]:



fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(0, 10, 0.1)
y = stats['train_loss']
z = stats['val_loss']
ax.plot(y, color='blue', label='train_loss')
ax.plot(z, color='grey', label='val_loss')
ax.set_xlabel('epochs')
ax.set_ylabel('loss value')
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(0, 10, 0.1)
z = stats['val_loss']
ax.plot(z, color='grey', label='val_loss')
ax.set_xlabel('epochs')
ax.set_ylabel('loss value')
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
z = stats['val_acc']
ax.plot(z, color='grey', label='val_acc')
ax.set_xlabel('epochs')
ax.set_ylabel('acc value')
ax.legend()
plt.show()

