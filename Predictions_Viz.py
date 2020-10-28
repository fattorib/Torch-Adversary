import matplotlib.pyplot as plt
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import numpy as np


def predict_class(X, Y, model):
    # model.cuda()

    class_names = ['Zero', 'One', 'Two',
                   'Three', 'Four', 'Five',
                   'Six', 'Seven', 'Eight', 'Nine']
    # X = X.cuda()

    logpred = model.forward(X)
    top_p, top_class = logpred.topk(1, dim=1)

    class_name = class_names[top_class[0].item()]

    class_prob = top_p[0].item()
    # .cpu() moves to cpu, .detach removes gradient tracking for specific tensor

    probabilities = torch.exp(logpred).cpu().detach().numpy()

    plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    plt.sca(ax1)
    plt.xticks([])
    plt.yticks([])

    probability = 100*max(probabilities[0])

    plt.title('{} \n Probability {:.2f} %'.format(class_name, probability))

    ax1.imshow(X[0].cpu().detach().permute(1, 2, 0), cmap='gray')

    plt.sca(ax2)
    plt.xticks(np.arange(10), ['Zero', 'One', 'Two',
                               'Three', 'Four', 'Five',
                               'Six', 'Seven', 'Eight', 'Nine'])

    ax2.bar(np.arange(10), probabilities[0], align='center')

    plt.show()
