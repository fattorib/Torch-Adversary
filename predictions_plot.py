import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def plot_predictions(X_tensor, model, cuda=True):
    """
    Take an input tensor, feed it through the model and output its predictions
    """

    unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225))
    # Get list of labels

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
    plt.imshow(unorm(X_tensor[0]).detach().cpu().permute(1, 2, 0))
    plt.show()
