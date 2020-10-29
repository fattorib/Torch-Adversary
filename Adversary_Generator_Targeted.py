from Predictions_Viz import predict_class
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


test_data = QMNIST(root='./QMNISTdata', train=False,
                   download=False, transform=transform)

# Define dataloaders
testloader = DataLoader(test_data, batch_size=1, shuffle=True)

# Define network
# Checking whether we have GPU
train_on_gpu = torch.cuda.is_available()

# Whether to retrain model or use precomputed params
train_model = True


if(train_on_gpu):
    print('GPU available.')
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


image_test, label_test = next(iter(testloader))

label_idx = label_test[0].item()

image_test.requires_grad = True

criterion = nn.NLLLoss()

model = Model()
model.eval()
state_dict = torch.load('Trained_model.pth')
model.load_state_dict(state_dict)

predict_class(image_test, label_test, model)


def create_FGSM(X, Y, epsilon):

    X.requires_grad = True

    model.zero_grad()
    output = model(X)

    loss = criterion(output, Y.view(1))

    loss.backward()

    X_adv = X + epsilon*(X.grad.sign())

    return torch.clamp(X_adv,  0, 1), epsilon*(X.grad.sign())


def create_adversary(X, Y_target, lr, epochs):
    X.requires_grad = True
    optimizer = torch.optim.SGD([X], lr=lr)
# Pass tensors to GPU. Weird issue with gradient tracking when passing from cpu to gpu...
    for e in range(1, epochs+1):

        # images, labels = images.cuda(), label.cuda()
        # images.requires_grad = True
        running_loss = 0

        optimizer.zero_grad()

        output = model(X)

        loss = criterion(output, Y_target.view(1))

        running_loss += loss.item()

        loss.backward()

        optimizer.step()

        if e % 100 == 0:
            print('Loss:', running_loss)
    return X


label_test = torch.tensor(2)
X_adv = create_adversary(image_test, label_test, 0.0005, 5000)

predict_class(X_adv, label_test, model)
