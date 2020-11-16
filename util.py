import os

import torch
import torch.nn as nn


def conv(c_in, c_out, k_size, stride=2, pad=1, bias=False, norm='bn', activation=None):
    layers = []

    # Conv.
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=bias))

    # Normalization
    if norm == 'bn':
        layers.append(nn.BatchNorm2d(c_out))
    elif norm == 'in':
        layers.append(nn.InstanceNorm2d(c_out))
    elif norm == None:
        pass

    # Activation
    if activation == 'lrelu':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif activation == None:
        pass

    return nn.Sequential(*layers)


def deconv(c_in, c_out, k_size, stride=2, pad=1, output_padding=0, bias=False, norm='bn', activation=None):
    layers = []

    # Deconv.
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, output_padding, bias=bias))

    # Normalization
    if norm == 'bn':
        layers.append(nn.BatchNorm2d(c_out))
    elif norm == 'in':
        layers.append(nn.InstanceNorm2d(c_out))
    elif norm == None:
        pass

    # Activation
    if activation == 'lrelu':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif activation == None:
        pass

    return nn.Sequential(*layers)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def save_checkpoint(model, save_path, device):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.to(device)


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print("Invalid path!")
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
