import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if 'test' in self.img_path[index]:
            return img, torch.from_numpy(np.array(0))
        else:
            label = self.img_label[index]
            ls = torch.from_numpy(np.array([float(x) for x in label.split(' ')]))
            return img, ls.float()

    def __len__(self):
        return len(self.img_path)
