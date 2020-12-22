# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # ReLU, tanh
from torch.utils.data import DataLoader  # handy to get mini batches
import torchvision.datasets as datasets  # datasets from pytorch
import torchvision.transforms as transforms  # stuff to transform the datasets. dont know


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28
seq_length = 28  # images are 1x28x28 -> interpret the first 28 as a sequence (no sense)
num_layers = 2
hidden_size = 256  # number of nodes in hidden layer
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2
filename_checkpoints = "checkpoint.pth.tar"
load_model = False


# Create a bidirectional RNN
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        #The init state h0 has shape [num_layers * num_directions, batch, hidden_size]
        # we are not using the next hidden_state (it is only been used within the very sequence)
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # normal rnn:
        # out, h_next = self.rnn(x, h0)

        # GRU:
        # out, h_next = self.gru(x, h0)

        # LSTM:
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)  # cell state for LSTM only
        out, h_next = self.lstm(x, (h0, c0))



        # shape of out normally: [seq_len, batch, num_directions * hidden_size]
        # With batch_first option: [batch, seq_length, num_directions * hidden_size]
        # out = out.reshape(out.shape[0], -1)
        # out = self.fc(out)
        out = self.fc(out[:, -1, :]) # Using only last hidden-state: all batches, last hidden state, and all features
        return out

def save_checkpoints(checkpoint, filename=filename_checkpoints):
    print("=> Saving checkpoints")
    torch.save(checkpoint, filename_checkpoints)

def load_checkpoints(checkpoint):
    print("=> Loading checkpoints")
    model.load_state_dict((checkpoint['state_dict']))
    optimizer.load_state_dict((checkpoint['optimizer_state_dict']))


# Testing
# good way of testing the nn, is to check if the output has the correct shape:
model = BRNN(input_size, hidden_size, num_layers, num_classes)  # Input will be 28x28=784, output-dim = 10
x = torch.randn(64, 28, 28)  # 64 images in your batch, 28 time_seqs, 28 features
print(model(x).shape)  # shape: (64, 10)




# Accuracy Method
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # dropout, batchnorm-layer behaving differently in eval
    with torch.no_grad():  # no_grad for saving memory for not creating the buferes in forward, needed for gradients
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.squeeze(1)

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
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# shuffle: suffle the images for each epoch
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Init network
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoints(torch.load(filename_checkpoints))

# Train network
for epoch in range(num_epochs):
    # saving checkpoints every 3 epochs
    if epoch % 3 == 0 and epoch > 0:
        checkpoint = {
            'state_dict': model.state_dict(),  # the weights
            'optimizer_state_dict': optimizer.state_dict()  # contains buffers and parameters
        }
        save_checkpoints(checkpoint)

    # for calculating the mean loss
    epoch_loss = 0  # what this line does

    for batch_idx, (data, targets) in enumerate(train_loader):
        # data is tensor
        data = data.to(device=device)
        targets = targets.to(device=device)

        # get correct shape
        # print(data.shape)
        # shape: [64, 1, 28, 28] -> get rid of the second dim, we need: [batch, time_seq, features]
        data = data.squeeze(1)

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

