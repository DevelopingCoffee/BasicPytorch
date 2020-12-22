from torch.utils.data import Dataset
import torch
from skimage import io
import pandas as pd
import os


class DataloaderCatsAndDogs(Dataset):
    def __init__(self, csv_file, root_dir, image_folder, transform=None):
        self.annotations = pd.read_csv(os.path.join(root_dir, csv_file))
        self.image_dir = os.path.join(root_dir, image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        return (image, label)
