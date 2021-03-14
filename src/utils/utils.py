import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.transforms as transforms
from numpy.core.numeric import Infinity


def initialize_parameters(m, std=0.001):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, std)
        nn.init.constant_(m.bias.data, 0.0)


def initialize_parameters_kaiming(m, scale=0.1):
    classname = m.__class__.__name__
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def inverse_normalize(tensor, mean, std):
    return NormalizeInverse(mean, std)(tensor)


def normalize(tensor, mean, std):
    return transforms.Normalize(mean, std)(tensor)


def load_checkpoint(model, opt_e, opt_g, opt_d, sch_e, sch_g, ech_d, filename='net.pth'):
    start_epoch = 0
    logs = list()
    max_ssim = -Infinity
    max_ssim_epoch = 0
    bit_size = 0

    if os.path.isfile(filename):
        print("=> Loading checkpoint '{}'".format(filename))
        # state = torch.load(filename)
        state = None
        if torch.cuda.is_available():
            state = torch.load(filename)
        else:
            state = torch.load(filename, map_location=torch.device('cpu'))

        start_epoch = state['epoch']
        logs = state['logs']
        bit_size = state['bit']
        model.load_state_dict(state['model_dict'])
        # optimizer
        opt_e.load_state_dict(state['optimizer_e'])
        opt_g.load_state_dict(state['optimizer_g'])
        opt_d.load_state_dict(state['optimizer_d'])
        # scheduler
        sch_e.load_state_dict(state['scheduler_e'])
        sch_g.load_state_dict(state['scheduler_g'])
        ech_d.load_state_dict(state['scheduler_d'])
    else:
        print("=> No checkpoint found at '{}'".format(filename))
        exit()

    return start_epoch, model, opt_e, opt_g, opt_d, sch_e, sch_g, ech_d, logs, max_ssim, max_ssim_epoch, bit_size


def add_padding(tensor, p):
    b, c, h, w = tensor.shape

    if h % p == 0 and w % p == 0 and (h == w):
        return tensor, h, w

    mx = max((math.ceil(h / (p + 0.0)) * p), (math.ceil(w / (p + 0.0)) * p))

    padding_h = max(0, mx - h)
    padding_w = max(0, mx - w)

    tensor = F.pad(tensor, [0, padding_w, 0, padding_h], 'reflect')
    return tensor, h, w
