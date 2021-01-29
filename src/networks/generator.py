from src.utils.tensor import save_img
from numpy.core.numeric import identity
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
            kernel[x + y * downscale_factor::downscale_factor *
                   downscale_factor, 0, y, x] = 1
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


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels,
                                kernel_size, stride)  # , padding)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            # self.upsample = nn.Upsample(scale_factor=upsample, mode='bicubic', align_corners=True)
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        norm_kwargs = dict(momentum=0.1, affine=True,
                           track_running_stats=False)

        # channel.ChannelNorm2D_wrap
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.BatchNorm2d(channels, **norm_kwargs)

        self.relu = nn.ReLU()

        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.BatchNorm2d(channels, **norm_kwargs)

    def forward(self, x):
        identity = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))

        out += identity
        out = self.relu(out)

        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # nonlineraity
        self.relu = nn.PReLU()
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        norm_kwargs = dict(momentum=0.1, affine=True,
                           track_running_stats=False)

        # self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.unshuffle = PixelUnshuffle(2)

        self.conv_1 = ConvLayer(12, 128, kernel_size=3, stride=1)

        # self.conv_2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        # self.in2_e = channel.ChannelNorm2D_wrap(64, **norm_kwargs)

        # self.conv_3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        # self.in3_e = channel.ChannelNorm2D_wrap(128, **norm_kwargs)

        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.res6 = ResidualBlock(128)
        self.res7 = ResidualBlock(128)
        self.res8 = ResidualBlock(128)
        self.res9 = ResidualBlock(128)

        self.deconv_4 = UpsampleConvLayer(128, 128, kernel_size=3, stride=1)
        self.in4_d = nn.BatchNorm2d(128, **norm_kwargs)

        self.deconv_3 = UpsampleConvLayer(
            128, 64, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.BatchNorm2d(64, **norm_kwargs)

        # self.deconv_2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        # self.in2_d = channel.ChannelNorm2D_wrap(32, **norm_kwargs)

        self.deconv_1 = UpsampleConvLayer(64, 3, kernel_size=3, stride=1)
        # self.in1_d = channel.ChannelNorm2D_wrap(3, **norm_kwargs)

    def forward(self, x):
        # Recovery from encoder
        # (3, 256, 268) -> (12, 128, 128)
        y = self.unshuffle(x)

        # (3, 256, 256) -> (3, 512, 512)
        # y = self.upsample(y)

        y = self.leakyRelu(self.conv_1(y))
        # y = self.relu(self.in2_e(self.conv_2(y)))
        # y = self.relu(self.in3_e(self.conv_3(y)))

        residual = y
        res = self.res1(y)
        res = self.res2(res)
        res = self.res3(res)
        res = self.res4(res)
        res = self.res5(res)
        res = self.res6(res)
        res = self.res7(res)
        res = self.res8(res)
        res = self.res9(res)

        res = self.in4_d(self.deconv_4(res))

        res = res + residual
        y = self.leakyRelu(res)

        y = self.relu(self.in3_d(self.deconv_3(y)))
        # y = self.relu(self.in2_d(self.deconv_2(y)))
        out = self.tanh(self.deconv_1(y))
        # y = self.conv_1(y)
        # y = self.conv_2(y)
        # y = self.conv_3(y)

        # residual = y
        # res = None
        # for m in range(self.n_residual_blocks):
        #     resblock_m = getattr(self, f'resblock_{str(m)}')
        #     if m == 0:
        #         res = resblock_m(y)
        #     else:
        #         res = resblock_m(res)

        # res = self.deconv_4(res)
        # res = res + residual
        # y = self.leakyRelu(res)

        # y = self.deconv_3(y)
        # y = self.deconv_2(y)
        # out = self.deconv_1(y)

        return out


def trial():
    x = torch.randn((2, 3, 256, 256))

    G = Generator()
    x_generated = G(x)

    print(x.size())
    print(x_generated.size())

    save_img(x[0], 'test.png')
    save_img(x_generated[0].detach(), 'test_gen.png')
