import torch
import torch.nn as nn
from torch.nn import init
from functools import partial

# import lpips

from src.utils.compression import _compress

from src.networks.encoder import Encoder
from src.networks.generator import Generator
from src.networks.discriminator import Discriminator
from src.networks.discriminator_hf import DiscriminatorHF

from src.losses.gan_loss import GANLoss
from src.losses.perceptual import VGGLoss
from src.losses.hf_losses import gan_loss
# from loader import normalize


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super(RMSELoss, self).__init__()

        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y) + self.eps)


class Model(nn.Module):
    def __init__(self, bit=3, opt=None):
        super(Model, self).__init__()

        self.Encoder = Encoder(cuda=opt.cuda, alpha=opt.a)
        self.Generator = Generator()
        self.Discriminator = DiscriminatorHF()

        self.gan_loss = GANLoss(cuda=opt.cuda)
        self.gan_loss_hf = partial(gan_loss, 'non_saturating')
        self.squared_difference = nn.MSELoss()

        self.bit_size = bit

    def compress(self, x):
        return _compress(x, self.bit_size)

    def compression_forward_eval(self, x):
        x = self.Encoder(x)
        compressed = _compress(x, self.bit_size)

        return compressed

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def distortion_loss(self, x_gen, x_real):
        return self.squared_difference(x_gen, x_real)  # / 255.

        # return sq_err

    def restruction_loss(self, reconstruction, input_image):
        x_real = input_image
        x_gen = reconstruction

        distortion_loss = self.distortion_loss(x_gen, x_real)

        return distortion_loss

    def compression_loss(self, reconstruction, input_image):
        x_real = input_image
        x_gen = reconstruction

        distortion_loss = self.distortion_loss(x_gen, x_real)

        return distortion_loss
