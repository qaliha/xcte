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


class Generator(nn.Module):
    def __init__(self, n_blocks=6, n_feature=64):
        super(Generator, self).__init__()

        # self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)

        model_conv_ = [PixelUnshuffle(2)]
        model_conv_ += [nn.Upsample(scale_factor=2, mode='nearest')]
        model_conv_ += [ConvLayer(12, n_feature, 9, 1,
                                  norm='none', activation='leaky')]

        self.model_conv = nn.Sequential(*model_conv_)

        model_resblocks_ = []
        for i in range(n_blocks):
            model_resblocks_ += [ResidualLayer(n_feature, n_feature, 3, 1)]

        self.model_resblocks = nn.Sequential(*model_resblocks_)

        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.model_resout = ConvLayer(
            n_feature, n_feature, 3, 1, activation='skip', norm='none')

        model_deconv_ = [ConvLayer(n_feature, int(
            n_feature / 2), 3, 1, norm='none')]
        model_deconv_ += [ConvLayer(int(n_feature / 2),
                                    3, 9, 1, activation='tanh')]

        self.model_deconv = nn.Sequential(*model_deconv_)

    def forward(self, x):
        y = self.model_conv(x)

        residual = y
        res = self.model_resblocks(y)
        res = self.model_resout(res)
        res = torch.add(res, residual)
        y = self.leaky(res)

        out = self.model_deconv(y)

        return out


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, activation='prelu', norm='channel'):
        super(ConvLayer, self).__init__()

        # padding
        self.pad = nn.ReflectionPad2d(kernel_size//2)

        # convolution
        self.conv_layer = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride)

        # activation
        if activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.activation = None

        # normalization
        if norm == 'channel':
            self.normalization = channel.ChannelNorm2D_wrap(out_ch,
                                                            momentum=0.1, affine=True, track_running_stats=False)
        elif norm == 'batch':
            self.normalization = nn.BatchNorm2d(out_ch)
        else:
            self.normalization = None

    def forward(self, x):
        x = self.pad(x)
        x = self.conv_layer(x)
        if self.normalization is not None:
            x = self.normalization(x)

        if self.activation is not None:
            # if activation is not none call the activation
            x = self.activation(x)

        return x


class ResidualLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(ResidualLayer, self).__init__()

        self.prelu = nn.PReLU()
        self.conv1 = ConvLayer(in_ch, out_ch, kernel_size,
                               stride)

        self.conv2 = ConvLayer(out_ch, out_ch, kernel_size,
                               stride, norm='skip', activation='none')

    def forward(self, x):
        identity_map = x
        res = self.conv1(x)
        res = self.conv2(res)

        res = torch.add(res, identity_map)
        out = self.prelu(res)

        return out


class DeconvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, activation='relu', upsample='nearest', norm='batch'):
        super(DeconvLayer, self).__init__()

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample)

        # pad
        self.pad = nn.ReflectionPad2d(kernel_size//2)

        # conv
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)

        # activation
        if activation == 'prelu':
            self.activation = nn.PReLU()
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError("Not implemented!")

        # normalization
        if norm == 'channel':
            self.normalization = channel.ChannelNorm2D_wrap(out_ch,
                                                            momentum=0.1, affine=True, track_running_stats=False)
        elif norm == 'batch':
            self.normalization = nn.BatchNorm2d(out_ch)
        else:
            self.normalization = None

    def forward(self, x):
        x = self.upsample(x)
        x = self.pad(x)
        x = self.conv(x)
        if self.normalization is not None:
            x = self.normalization(x)
        x = self.activation(x)
        return x


def trial():
    x = torch.randn((2, 3, 256, 256))

    G = Generator()
    x_generated = G(x)

    print(x.size())
    print(x_generated.size())

    save_img(x[0], 'interm/test.png')
    save_img(x_generated[0].detach(), 'interm/test_gen.png')
