import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def UnNormalize(mean, std, tensor):
    """
    Unnormalize an image for plotting
    """
    tensor_copy = tensor.clone()
    for t, m, s in zip(tensor_copy, mean, std):
        t.mul_(s).add_(m)

    return tensor_copy


def eval_tensor(X_tensor, model, cuda):
    """
    X_tensor: Single tensor of dimension [1,3,224,224]
    model: pretrained model we are evaluating on
    cuda: bool

    returns predicted label, probability
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

    return value, pred_prob


def plot_predictions_subplot(X_tensor_original, X_tensor_adv, model, cuda):
    """
    Plot image and its adversary along with probabilities/predicted classes.
    """
    json_file = open('labels.json')
    labels_json = json.load(json_file)
    labels = {int(idx): label for idx, label in labels_json.items()}

    # First image of batch
    label_idx_original, pred_prob_original = eval_tensor(
        X_tensor_original[0].unsqueeze(0), model, cuda)

    label_idx_adv, pred_prob_adv = eval_tensor(
        X_tensor_adv[0].unsqueeze(0), model, cuda)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_original = UnNormalize(mean, std, X_tensor_original)
    image_adv = UnNormalize(mean, std, X_tensor_adv)

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))

    plt.sca(axs[0, 0])
    plt.xticks([])
    plt.yticks([])
    plt.title('{} \n Probability {:.2f} %'.format(
        labels[label_idx_original], pred_prob_original))

    plt.imshow(image_original[0].detach().cpu().permute(1, 2, 0))

    plt.sca(axs[0, 1])
    plt.xticks([])
    plt.yticks([])
    plt.title('{} \n Probability {:.2f} %'.format(
        labels[label_idx_adv], pred_prob_adv))

    plt.imshow(image_adv[0].detach().cpu().permute(1, 2, 0))

    # Second Image of batch
    label_idx_original, pred_prob_original = eval_tensor(
        X_tensor_original[1].unsqueeze(0), model, cuda)

    label_idx_adv, pred_prob_adv = eval_tensor(
        X_tensor_adv[1].unsqueeze(0), model, cuda)

    plt.sca(axs[1, 0])
    plt.xticks([])
    plt.yticks([])
    plt.title('{} \n Probability {:.2f} %'.format(
        labels[label_idx_original], pred_prob_original))

    plt.imshow(image_original[1].detach().cpu().permute(1, 2, 0))

    plt.sca(axs[1, 1])
    plt.xticks([])
    plt.yticks([])
    plt.title('{} \n Probability {:.2f} %'.format(
        labels[label_idx_adv], pred_prob_adv))

    plt.imshow(image_adv[1].detach().cpu().permute(1, 2, 0))

    # Third image of batch
    label_idx_original, pred_prob_original = eval_tensor(
        X_tensor_original[2].unsqueeze(0), model, cuda)

    label_idx_adv, pred_prob_adv = eval_tensor(
        X_tensor_adv[2].unsqueeze(0), model, cuda)

    plt.sca(axs[2, 0])
    plt.xticks([])
    plt.yticks([])
    plt.title('{} \n Probability {:.2f} %'.format(
        labels[label_idx_original], pred_prob_original))

    plt.imshow(image_original[2].detach().cpu().permute(1, 2, 0))

    plt.sca(axs[2, 1])
    plt.xticks([])
    plt.yticks([])
    plt.title('{} \n Probability {:.2f} %'.format(
        labels[label_idx_adv], pred_prob_adv))

    plt.imshow(image_adv[2].detach().cpu().permute(1, 2, 0))

    plt.show()
