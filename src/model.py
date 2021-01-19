import torch
import torch.nn as nn
from torch.nn import init

from src.utils.compression import compress

from src.networks.encoder import Encoder
from src.networks.generator import Generator
from src.networks.discriminator import Discriminator

from src.losses.gan_loss import GANLoss
from src.losses.perceptual import VGGLoss

class Model(nn.Module):
    def __init__(self, bit=3, opt=None):
        super(Model, self).__init__()

        self.Encoder = Encoder(cuda=opt.cuda)
        self.Generator = Generator()
        self.Discriminator = Discriminator()

        self.gan_loss = GANLoss(cuda=opt.cuda)
        self.squared_difference = torch.nn.MSELoss(reduction='mean')
        self.perceptual_loss = VGGLoss()

        self.__initialize_weights(self.Encoder)
        self.__initialize_weights(self.Generator)
        self.__initialize_weights(self.Discriminator)

        self.bit_size = bit

    def __initialize_weights(self, net):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

        net.apply(init_func)

    def compression_forward_eval(self, x):
        with torch.no_grad():
            x = self.Encoder(x)
            compressed = compress(x, self.bit_size)

        return compressed

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

    def e_train(self, original):
        x = self.Encoder(original)
        x = compress(x, self.bit_size)

        x = self.Generator(x)

        compression_losses = self.squared_difference(x, original)
        return compression_losses

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
