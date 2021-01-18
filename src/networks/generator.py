import torch
import torch.nn as nn
import torch.nn.functional as F

from src.norm import channel

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''

        return pixel_unshuffle(input, self.downscale_factor)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.relu = nn.ReLU()
        self.norm = channel.ChannelNorm2D_wrap

        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(3//2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1),
            self.norm(channels),
            self.relu
        )

        self.conv_2 = nn.Sequential(
            nn.ReflectionPad2d(3//2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1),
            self.norm(channels),
        )

    def forward(self, x):
        identity = x
        y = self.conv_1(x)
        y = self.conv_2(y)
        y = y + identity

        y = self.relu(y)

        return y

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.n_residual_blocks = 9

        # nonlineraity
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        self.norm = channel.ChannelNorm2D_wrap
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.unshuffle = PixelUnshuffle(2)

        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(9//2),
            nn.Conv2d(12, 32, kernel_size=9, stride=1),
            self.norm(32, **norm_kwargs),
            self.relu
        )

        self.conv_2 = nn.Sequential(
            nn.ReflectionPad2d(3//2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            self.norm(64, **norm_kwargs),
            self.relu
        )

        self.conv_3 = nn.Sequential(
            nn.ReflectionPad2d(3//2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            self.norm(128, **norm_kwargs),
            self.relu
        )

        for m in range(self.n_residual_blocks):
            resblock_m = ResidualBlock(128)
            self.add_module(f'resblock_{str(m)}', resblock_m)

        # Decoding layers
        self.deconv_4 = nn.Sequential(
            nn.ReflectionPad2d(3//2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            self.norm(128),
        )

        self.deconv_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(3//2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            self.norm(64, **norm_kwargs),
            self.relu
        )

        self.deconv_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(3//2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            self.norm(32, **norm_kwargs),
            self.relu
        )

        self.deconv_1 = nn.Sequential(
            nn.ReflectionPad2d(9//2),
            nn.Conv2d(32, 3, kernel_size=9, stride=1),
            self.norm(3, **norm_kwargs),
            self.tanh
        )

    def forward(self, x):
        # Recovery from encoder
        # (3, 256, 256) -> (3, 512, 512)
        head = self.upsample(x)
        # (3, 512, 512) -> (12, 256, 256)
        head = self.unshuffle(head)

        head = self.conv_1(head)
        head = self.conv_2(head)
        head = self.conv_3(head)

        for m in range(self.n_residual_blocks):
            resblock_m = getattr(self, f'resblock_{str(m)}')
            if m == 0:
                x = resblock_m(head)
            else:
                x = resblock_m(x)

        x = self.deconv_4(x)
        x = x + head
        x = self.leakyRelu(x)

        x = self.deconv_3(x)
        x = self.deconv_2(x)
        out = self.deconv_1(x)

        return out

def trial():
    x = torch.randn((2, 3, 256, 256))

    G = Generator()
    x_generated = G(x)

    print(x.size())
    print(x_generated.size())