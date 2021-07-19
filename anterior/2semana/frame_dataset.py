import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image



class FrameDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        frame_list = []
        count = 0
        video_path = os.path.join(self.root_dir, self.info.iloc[index, 0])
        for frame in os.listdir(video_path):
            frame_path = os.path.join(video_path, frame)
            image = Image.open(frame_path).convert('RGB')
            frame_list.insert(0, image)
            count = count + 1
            if count == 10:
                break
            # TODO: fazer uma lista de frames
        label = self.info.iloc[index, 1]

        if self.transform:
            for image in frame_list:
                image = self.transform(image)
                frame_list.pop()
                frame_list.insert(0, image)

        return frame_list, label
