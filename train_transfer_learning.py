# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # ReLU, tanh
from torch.utils.data import DataLoader  # handy to get mini batches
import torchvision.datasets as datasets  # datasets from pytorch
import torchvision.transforms as transforms  # stuff to transform the datasets. dont know
import torchvision


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10
filename_checkpoints = "checkpoint.pth.tar"
load_model = False

# pretrained models from pytorch are available on:
# https://pytorch.org/docs/stable/torchvision/models.html

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# load pretrained and modify it:
model = torchvision.models.vgg16(pretrained=True) # if pretrained=False only architecture is used

# disable training for all existing layers (as already pretrained)
for param in model.parameters():
    param.requires_grad = False


# task: remove the last two layers, the averagepooling and the sequential classifier.
# The new ones are requires_grad = True
model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 10))
model.to(device)

# print(model)
# import sys
# sys.exit()


# Testing
# good way of testing the nn, is to check if the output has the correct shape:
# model = CNN(in_channels=1, num_classes=10) # black/white images
# x = torch.randn(64, 1, 28, 28)  # 64 images in your batch, 784 input dim
# print(model(x).shape)  # shape: (64, 10)


def save_checkpoints(checkpoint, filename=filename_checkpoints):
    print("=> Saving checkpoints")
    torch.save(checkpoint, filename_checkpoints)

def load_checkpoints(checkpoint):
    print("=> Loading checkpoints")
    model.load_state_dict((checkpoint['state_dict']))
    optimizer.load_state_dict((checkpoint['optimizer_state_dict']))



# Accuracy Method
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # dropout, batchnorm-layer behaving differently in eval
    with torch.no_grad():  # no_grad for saving memory for not creating the buferes in forward, needed for gradients
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            # scores-shape: [64, 10]
            scores = model(x)

            # max also squeezes dimension: pred_indices-shape: [64]
            val, pred_indices = torch.max(scores, dim=1)

            # y-shape: [64]
            num_correct += (pred_indices == y).sum()
            num_samples += pred_indices.size(dim=0)

        # print(f'Accuracy on {"train" if loader.dataset.train else "test" }-data: {float(num_correct)/float(num_samples):.2f}')
        model.train()
        return float(num_correct)/float(num_samples) * 100


# Load Data
# MNIST is in numpy, so transform it
# MNIST is a dataset of handwritten numbers
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# shuffle: suffle the images for each epoch
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

print(len(train_dataset))

# Init network
#model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoints(torch.load(filename_checkpoints))


# Train network
for epoch in range(num_epochs):

    # saving checkpoints every 3 epochs
    if epoch % 3 == 0 and epoch>0:
        checkpoint = {
            'state_dict': model.state_dict(),  # the weights
            'optimizer_state_dict': optimizer.state_dict()  # contains buffers and parameters
        }
        save_checkpoints(checkpoint)

    # for calculating the mean loss
    epoch_loss = 0  # what this line does

    # gets out the mini_batches
    for batch_idx, (data, targets) in enumerate(train_loader):

        # data is tensor
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        epoch_loss += loss.item()


        # backward (remember graph has automatically been build by forwarding, with it
        # the bufferes needed for back-propagation) -> calculating the gradients
        loss.backward()

        # updating the weights by using the gradients (from backward function). And set gradients back to zero
        optimizer.step()
        optimizer.zero_grad()
    acc = check_accuracy(train_loader, model)
    print(f"Epoch {epoch + 1} with accuracy {acc:.2f} and loss {epoch_loss/len(train_loader):.4f}")



check_accuracy(test_loader, model)

