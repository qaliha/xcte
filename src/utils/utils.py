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


def mkdir(directory, mode=0o777):
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.chmod(directory, mode=mode)


def dir_exists(directory):
    return os.path.exists(directory)


def apply_threshold(value):
    "Returns 0 or 255 depending where value is closer"
    return 255 * math.floor(value/128)


def sp_halftone(image_file):
    pixel = image_file.load()

    x_lim, y_lim = image_file.size

    for y in range(1, y_lim):
        for x in range(1, x_lim):
            red_oldpixel, green_oldpixel, blue_oldpixel = pixel[x, y]

            red_newpixel = apply_threshold(red_oldpixel)
            green_newpixel = apply_threshold(green_oldpixel)
            blue_newpixel = apply_threshold(blue_oldpixel)

            pixel[x, y] = red_newpixel, green_newpixel, blue_newpixel

            red_error = red_oldpixel - red_newpixel
            blue_error = blue_oldpixel - blue_newpixel
            green_error = green_oldpixel - green_newpixel

            if x < x_lim - 1:
                red = pixel[x+1, y][0] + round(red_error * 7/16)
                green = pixel[x+1, y][1] + round(green_error * 7/16)
                blue = pixel[x+1, y][2] + round(blue_error * 7/16)

                pixel[x+1, y] = (red, green, blue)

            if x > 1 and y < y_lim - 1:
                red = pixel[x-1, y+1][0] + round(red_error * 3/16)
                green = pixel[x-1, y+1][1] + round(green_error * 3/16)
                blue = pixel[x-1, y+1][2] + round(blue_error * 3/16)

                pixel[x-1, y+1] = (red, green, blue)

            if y < y_lim - 1:
                red = pixel[x, y+1][0] + round(red_error * 5/16)
                green = pixel[x, y+1][1] + round(green_error * 5/16)
                blue = pixel[x, y+1][2] + round(blue_error * 5/16)

                pixel[x, y+1] = (red, green, blue)

            if x < x_lim - 1 and y < y_lim - 1:
                red = pixel[x+1, y+1][0] + round(red_error * 1/16)
                green = pixel[x+1, y+1][1] + round(green_error * 1/16)
                blue = pixel[x+1, y+1][2] + round(blue_error * 1/16)

                pixel[x+1, y+1] = (red, green, blue)

    return image_file
