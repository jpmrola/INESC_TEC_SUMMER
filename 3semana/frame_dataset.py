import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
import random
from PIL import Image


class FrameDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.frames_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.slice_size = 10

    def __len__(self):
        return len(self.frames_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # need to take 10 frames and stack them on 1 tensor
        # ()-> second index has to be 1 because id column (3 columns)
        # ()-> why have to be NOT??? is wrong!! but works..
        all_frames = [f for f in os.listdir(os.path.join(self.root_dir, str(self.frames_csv.iloc[idx, 0]))) if
                      not os.path.isfile(f)]
        # random get 10 consecutive frames from all video frames in folder
        if len(all_frames) > (self.slice_size):
            start = random.randrange(len(all_frames) - self.slice_size)
            frames = all_frames[start: start + self.slice_size]
        else:
            frames = all_frames

        images = []
        # load the images
        for frame in frames:
            path = os.path.join(self.root_dir, str(self.frames_csv.iloc[idx, 0])) + '/' + frame
            images.append(Image.open(path).convert('RGB'))
        transform = transforms.Compose([
            transforms.Resize((240, 320), ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])])
        # apply tansforms and condense all the images in 1 tensor
        tensors = []
        for image in images:
            tensors.append(transform(image))
        final_image = torch.stack(tensors, dim=1)
        tag = int(self.frames_csv.iloc[idx, 1])
        return final_image, tag