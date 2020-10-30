import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def UnNormalize(mean, std, tensor):
    tensor_copy = tensor.clone()
    for t, m, s in zip(tensor_copy, mean, std):
        t.mul_(s).add_(m)

    return tensor_copy


def plot_predictions(X_tensor, model, cuda=True):
    """
    Take an input tensor, feed it through the model and output its predictions
    """
    json_file = open('labels.json')
    labels_json = json.load(json_file)
    labels = {int(idx): label for idx, label in labels_json.items()}

    if cuda == True:
        X_tensor = X_tensor.cuda()
        model.cuda()

    prediction = model(X_tensor)
    output_probs = F.softmax(prediction, dim=1)
    pred_prob = np.round(
        (torch.max(output_probs.cpu().data, 1)[0][0]) * 100, 4)

    label_idx = torch.max(prediction.data, 1)[1][0]
    value = label_idx.item()

    plt.xticks([])
    plt.yticks([])
    plt.title('{} \n Probability {:.2f} %'.format(labels[value], pred_prob))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image = UnNormalize(mean, std, X_tensor)
    plt.imshow(image[0].detach().cpu().permute(1, 2, 0))
    plt.show()
