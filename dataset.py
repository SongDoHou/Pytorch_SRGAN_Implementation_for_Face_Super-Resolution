import os
from torch.utils.data import Dataset
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch
import numpy as np

class FaceData(Dataset):
    def __init__(self):
        self.train_path = "" # Training Dataset Location
        self.train_img_list = os.walk(self.train_path).__next__()[2]
        self.len_dataset = len(self.train_img_list)
        self.transform_input = A.Compose([A.Resize(32, 32), A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()])
        self.transform_gt = A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()])

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.train_path, self.train_img_list[item]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return self.transform_input(image=img)['image'], self.transform_gt(image=img)['image']

if __name__ == "__main__":
    dataset = FaceData()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    for input_data, gt in data_loader:
        print(input_data.size())
        print(gt.size())
        exit()
