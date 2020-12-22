# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # ReLU, tanh
from torch.utils.data import DataLoader  # handy to get mini batches
import torchvision.datasets as datasets  # datasets from pytorch
import torchvision.transforms as transforms  # stuff to transform the datasets. dont know
import torchvision
from dataloader_helper import DataloaderCatsAndDogs

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 100
filename_checkpoints = "checkpoint.pth.tar"
load_model = False



transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0,0], std=[1.0, 1.0, 1.0]), # find for each channel first mean and std
])

dataset = DataloaderCatsAndDogs(csv_file='cats_dogs.csv', root_dir='/Users/dirk/development/ba/torch_tut.py/dataset/cats_and_dogs', image_folder='cats_dogs_resized', transform=transformations )
# in any epoch the dataloader will apply a fresh set of random operations
portions = [int(len(dataset)*0.8), int(len(dataset)*0.2)]  # split of dataset
train_set, test_set = torch.utils.data.random_split(dataset,portions)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

