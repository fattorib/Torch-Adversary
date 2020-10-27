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


if(train_on_gpu):
    print('Using GPU!')
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

state_dict = torch.load('Trained_model.pth')
model.load_state_dict(state_dict)

# Visualizing Predictions
class_names = ['Zero', 'One', 'Two',
               'Three', 'Four', 'Five',
               'Six', 'Seven', 'Eight', 'Nine']


images, labels = next(iter(testloader))

image = images.cuda()

logpred = model.forward(image)
top_p, top_class = logpred.topk(1, dim=1)


class_name = class_names[top_class[0].item()]

class_prob = top_p[0].item()
# .cpu() moves to cpu, .detach removes gradients for specific tensor

probabilities = torch.exp(logpred).cpu().detach().numpy()


plt.style.use('ggplot')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

plt.sca(ax1)
plt.xticks([])
plt.yticks([])

probability = 100*max(probabilities[0])

plt.title('{} \n Probability {:.2f} %'.format(class_name, probability))

ax1.imshow(images[0].permute(1, 2, 0), cmap='bone')


plt.sca(ax2)
plt.xticks(np.arange(10), ['Zero', 'One', 'Two',
                           'Three', 'Four', 'Five',
                           'Six', 'Seven', 'Eight', 'Nine'])

ax2.bar(np.arange(10), probabilities[0], align='center')

plt.show()
