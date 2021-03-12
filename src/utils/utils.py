import torch
import os
import torchvision.transforms as transforms
from numpy.core.numeric import Infinity


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


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
