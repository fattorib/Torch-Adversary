import matplotlib.pyplot as plt
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import numpy as np

from torchvision.datasets import QMNIST

transform = transforms.Compose([
    transforms.ToTensor()
])


# Load in datasets
train_data = QMNIST(root='./QMNISTdata', train=True,
                    download=False, transform=transform)

test_data = QMNIST(root='./QMNISTdata', train=False,
                   download=False, transform=transform)

# Define dataloaders
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
testloader = DataLoader(test_data, batch_size=64, shuffle=True)


# Define network
# Checking whether we have GPU
train_on_gpu = torch.cuda.is_available()

# Whether to retrain model or use precomputed params
train_model = False


if(train_on_gpu):
    print('Training on GPU!')
else:
    print('No GPU available, training on CPU; consider making n_epochs very small.')


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        # Tuple given as: input channels, feature maps, kernel size. Keeping default size of 1
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)

        # Depth here is the number of feature maps we had previously passed
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

        # Pooling layer with size of 2 and stride of 2. Halving the xy size.
        self.pool = nn.MaxPool2d(2, 2)

        # Final image size is x_final*y_final*depth
        self.fc1 = nn.Linear(7*7*32, 64)

        self.fc2 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))

        x = x.view(-1, 7*7*32)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


criterion = nn.NLLLoss()

model = Model()

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

model.cuda()

epochs = 4

if train_model == True:
    for e in range(1, epochs+1):

        running_loss = 0

        for images, labels in trainloader:

            # Pass tensors to GPU
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()

            output = model(images)

            loss = criterion(output, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

        if e % 2 == 0:

            # Evaluate model on test data
            accuracy = 0
            test_loss = 0
            # Turning off gradient tracking
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    logps = model.forward(inputs)

                    loss = criterion(logps, labels)
                    test_loss += loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print('Epoch:', e)
            print('Training Loss:', running_loss/len(trainloader))
            print('Test Loss:', test_loss/len(testloader))
            print('Test Accuracy:', 100*accuracy/len(testloader), '%')

            # Turning back on gradient tracking
            model.train()

    # Saving model parameters
    torch.save(model.state_dict(), 'Trained_model.pth')
