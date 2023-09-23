import numpy as np
import matplotlib.pyplot as plt
import torch

def saliency(input, model):
    # we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False

    # set model in eval mode
    model.eval()

    input.unsqueeze_(0)

    # we want to calculate gradient of highest score w.r.t. input
    # so set requires_grad to True for input
    input.requires_grad = True
    input = input.to(dtype=torch.float)
    input.retain_grad()
    # forward pass to calculate predictions
    preds = model(input)
    score, indices = torch.max(preds, 1)
    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    # get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    # normalize to [0..1]
    slc = (slc - slc.min()) / (slc.max() - slc.min())

    slc = slc.numpy()
    indices = indices.numpy()
    return slc, indices