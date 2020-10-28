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


# Noise image:
images = torch.rand(1, 1, 28, 28)
images.requires_grad = True


label = torch.tensor(3)

criterion = nn.NLLLoss()

model = Model()
state_dict = torch.load('Trained_model.pth')
model.load_state_dict(state_dict)

model
model.eval()

optimizer = torch.optim.SGD([images], lr=0.5)
epochs = 1000
# Pass tensors to GPU. Weird issue with gradient tracking when passing from cpu to gpu...
for e in range(1, epochs+1):

    # images, labels = images.cuda(), label.cuda()
    # images.requires_grad = True
    running_loss = 0

    optimizer.zero_grad()

    output = model(images)

    loss = criterion(output, label.view(1))

    running_loss += loss.item()

    loss.backward()

    optimizer.step()

    if e % 1000 == 0:
        print('Loss:', running_loss)


model.cuda()
image = images.cuda()

logpred = model.forward(image)
top_p, top_class = logpred.topk(1, dim=1)

class_names = ['Zero', 'One', 'Two',
               'Three', 'Four', 'Five',
               'Six', 'Seven', 'Eight', 'Nine']


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

ax1.imshow(image[0].cpu().detach().permute(1, 2, 0), cmap='bone')

plt.sca(ax2)
plt.xticks(np.arange(10), ['Zero', 'One', 'Two',
                           'Three', 'Four', 'Five',
                           'Six', 'Seven', 'Eight', 'Nine'])

ax2.bar(np.arange(10), probabilities[0], align='center')
plt.show()
