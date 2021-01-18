import torch
import torch.nn as nn

from src.utils.compression import compress

from src.networks.encoder import Encoder
from src.networks.generator import Generator
from src.networks.discriminator import Discriminator

from src.losses.gan_loss import GANLoss
from src.losses.perceptual import VGGLoss

class Model(nn.Module):
    def __init__(self, bit=3):
        super(Model, self).__init__()

        self.Encoder = Encoder()
        self.Generator = Generator()
        self.Discriminator = Discriminator()

        self.gan_loss = GANLoss()
        self.squared_difference = torch.nn.MSELoss(reduction='mean')
        self.perceptual_loss = VGGLoss()

        self.bit_size = bit

    def decompression_forward(self, x):
        self.Encoder.train()
        self.Generator.eval()
    
        hdr = x
        # Compression the image using Encoder (with gradients)
        x = self.Encoder(x)
        x = compress(x, self.bit_size)

        # expanded = None
        # with torch.no_grad():
        x = self.Generator(x)

        compression_losses = self.squared_difference(x, hdr)
        return compression_losses

    def compression_forward(self, x):
        # real_a = compressed, real_b = ground truth
        # Evaluate mode for compresison network
        self.Encoder.eval()
        self.Generator.train()

        hdr = x
        # Compress the input using Encoder
        with torch.no_grad():
            x = self.Encoder(x)
            x = compress(x, self.bit_size)
        
        expanded = self.Generator(x)

        # Update discriminator
        fake_ab = self.Discriminator(torch.cat((x, expanded), 1).detach())
        loss_d_fake = self.gan_loss(fake_ab, False)

        real_ab = self.Discriminator(torch.cat((x, hdr), 1))
        loss_d_real = self.gan_loss(real_ab, True)

        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        # Update generator
        fake_ab = self.Discriminator(torch.cat((x, expanded), 1))

        gan_losses = self.gan_loss(fake_ab, True)
        decoder_losses = self.squared_difference(expanded, hdr)
        perceptual_losses = self.perceptual_loss(expanded, hdr)

        generator_losses = (gan_losses + decoder_losses + perceptual_losses)

        return discriminator_loss, generator_losses

    # X is real image
    def forward(self, x):
        pass
