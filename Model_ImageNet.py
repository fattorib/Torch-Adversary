import matplotlib.pyplot as plt
import torch
import torchvision
import requests
import io
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from predictions_plot import plot_predictions, plot_predictions_subplot

# Import model
from torchvision.models import resnet34
if torch.cuda.is_available():
    print('GPU available.')


model = resnet34(pretrained=True)

# Whatever you do, you do not want to track these gradients
model.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


test_images = datasets.ImageFolder('./Images', transform=transform)

testloader = DataLoader(test_images, batch_size=1, shuffle=True)


image_test, label = next(iter(testloader))

criterion = torch.nn.CrossEntropyLoss()


def create_FGSM(X, Y, epsilon):

    X.requires_grad = True

    model.zero_grad()
    output = model(X)

    loss = criterion(output, Y.view(1))

    loss.backward()

    X_adv = X + epsilon*(X.grad.sign())

    return X_adv, epsilon*(X.grad.sign())


def create_adversary(X, Y_target, lr, epochs):

    X = X.cuda()
    Y_target = Y_target.cuda()
    model.cuda()

    X.requires_grad = True
    optimizer = torch.optim.SGD([X], lr=lr)
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

        if e % 1000 == 0:
            print('Loss:', running_loss)
    return X


# Label for pay-phone
label_test = torch.tensor(707)

X_adv = create_adversary(image_test, label_test, 0.1, 1000)

plot_predictions_subplot(image_test, X_adv, model, cuda=True)
