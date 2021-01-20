import torch
import torch.nn as nn
from torch.nn import init

import lpips

from src.utils.compression import compress

from src.networks.encoder import Encoder
from src.networks.generator import Generator
from src.networks.discriminator import Discriminator

from src.losses.gan_loss import GANLoss
from src.losses.perceptual import VGGLoss

from loader import normalize
class Model(nn.Module):
    def __init__(self, bit=3, opt=None):
        super(Model, self).__init__()

        self.Encoder = Encoder(cuda=opt.cuda)
        self.Generator = Generator()
        self.Discriminator = Discriminator()

        self.gan_loss = GANLoss(cuda=opt.cuda)
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.perceptual_loss = VGGLoss()

        self.__initialize_weights(self.Encoder)
        self.__initialize_weights(self.Generator)
        self.__initialize_weights(self.Discriminator)

        self.bit_size = bit

        self.k_M = 0.075 * 2**(-5)
        self.k_P = 1.
        self.beta = 0.15

        self.loss_fn_alex = lpips.LPIPS(net='alex')

    def __initialize_weights(self, net):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

        net.apply(init_func)

    def compress(self, x):
        return compress(x, self.bit_size)

    def compression_forward_eval(self, x):
        x = self.Encoder(x)
        compressed = compress(x, self.bit_size)

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
        # loss in [0,255] space but normalized by 255 to not be too big
        # - Delegate scaling to weighting
        sq_err = self.squared_difference(x_gen*255., x_real*255.) # / 255.
        return torch.mean(sq_err)

    def perceptual_loss(self, pred, target, normalize = True):
        if normalize:
            target = 2 * target  - 1
            pred = 2 * pred  - 1
        
        perp_loss = self.loss_fn_alex(target, pred)
        return torch.mean(perp_loss)

    def compression_loss(self, reconstruction, input_image):
        x_real = input_image
        x_gen = reconstruction

        # Normalize the input image
        # [-1., 1.] -> [0., 1.]
        x_real = (x_real + 1.) / 2.
        x_gen = (x_gen + 1.) / 2

        distortion_loss = self.distortion_loss(x_gen, x_real)
        perceptual_loss = self.perceptual_loss(x_gen, x_real, normalize=True)
        print(distortion_loss)
        print(perceptual_loss)
        weighted_distortion = distortion_loss * self.k_M
        weighted_perceptual = perceptual_loss * self.k_P

        return weighted_distortion + weighted_perceptual

    # def gd_training(self, compressed, original):
    #     # Compressed = real_a, Original = real_b
    #     expanded = self.Generator(compressed)

    #     # Update discriminator
    #     fake_ab = self.Discriminator(torch.cat((compressed, expanded), 1).detach())
    #     loss_d_fake = self.gan_loss(fake_ab, False)

    #     real_ab = self.Discriminator(torch.cat((compressed, original), 1))
    #     loss_d_real = self.gan_loss(real_ab, True)

    #     discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

    #     # Update generator
    #     fake_ab = self.Discriminator(torch.cat((compressed, expanded), 1))

    #     gan_losses = self.gan_loss(fake_ab, True)
    #     decoder_losses = self.squared_difference(expanded, original)
    #     perceptual_losses = self.perceptual_loss(expanded, original)

    #     generator_losses = gan_losses + decoder_losses + perceptual_losses

    #     return discriminator_loss, generator_losses

    # def e_train(self, original):
    #     x = self.Encoder(original)
    #     x = compress(x, self.bit_size)
        
    #     # Normalize the output first
    #     x = normalize(x)
    #     x = self.Generator(x)

    #     compression_losses = self.squared_difference(x, original)
    #     return compression_losses

    # def decompression_forward(self, x):
    #     self.Encoder.train()
    #     self.Generator.eval()
    
    #     hdr = x
    #     # Compression the image using Encoder (with gradients)
    #     x = self.Encoder(x)
    #     x = compress(x, self.bit_size)

    #     # expanded = None
    #     # with torch.no_grad():
    #     x = self.Generator(x)

    #     compression_losses = self.squared_difference(x, hdr)
    #     return compression_losses

    # def compression_forward(self, x):
    #     # real_a = compressed, real_b = ground truth
    #     # Evaluate mode for compresison network
    #     self.Encoder.eval()
    #     self.Generator.train()

    #     hdr = x
    #     # Compress the input using Encoder
    #     with torch.no_grad():
    #         x = self.Encoder(x)
    #         x = compress(x, self.bit_size)
        
    #     expanded = self.Generator(x)

    #     # Update discriminator
    #     fake_ab = self.Discriminator(torch.cat((x, expanded), 1).detach())
    #     loss_d_fake = self.gan_loss(fake_ab, False)

    #     real_ab = self.Discriminator(torch.cat((x, hdr), 1))
    #     loss_d_real = self.gan_loss(real_ab, True)

    #     discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

    #     # Update generator
    #     fake_ab = self.Discriminator(torch.cat((x, expanded), 1))

    #     gan_losses = self.gan_loss(fake_ab, True)
    #     decoder_losses = self.squared_difference(expanded, hdr)
    #     perceptual_losses = self.perceptual_loss(expanded, hdr)

    #     generator_losses = (gan_losses + decoder_losses + perceptual_losses)

    #     return discriminator_loss, generator_losses

    # # X is real image
    # def forward(self, x):
    #     pass
