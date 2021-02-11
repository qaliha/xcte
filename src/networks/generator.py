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
    def __init__(self, n_blocks=6, in_feat=32):
        super(Generator, self).__init__()

        # self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)

        model_conv_ = [PixelUnshuffle(2)]
        model_conv_ += [nn.UpsamplingNearest2d(scale_factor=2)]
        model_conv_ += [ConvLayer(12, in_feat, 9, 1)]
        model_conv_ += [ConvLayer(in_feat, in_feat * 2, 3, 2)]
        model_conv_ += [ConvLayer(in_feat * 2, in_feat * 4, 3, 2)]

        self.model_conv = nn.Sequential(*model_conv_)

        model_resblocks_ = []
        for i in range(n_blocks):
            model_resblocks_ += [ResidualLayer(in_feat * 4, in_feat * 4, 3, 1)]

        self.model_resblocks = nn.Sequential(*model_resblocks_)

        model_deconv_ = [DeconvLayer(in_feat * 4, in_feat * 2, 3, 1)]
        model_deconv_ += [DeconvLayer(in_feat * 2, in_feat, 3, 1)]
        model_deconv_ += [ConvLayer(in_feat, 3, 9, 1, activation='tanh')]

        self.model_deconv = nn.Sequential(*model_deconv_)

    def forward(self, x):
        head = self.model_conv(x)
        x = self.model_resblocks(head)
        x += head
        out = self.model_deconv(x)

        return out


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, activation='prelu'):
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
        else:
            self.activation = lambda x: x

        # normalization
        self.normalization = channel.ChannelNorm2D_wrap(out_ch,
                                                        momentum=0.1, affine=True, track_running_stats=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv_layer(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class ResidualLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(ResidualLayer, self).__init__()

        self.conv1 = ConvLayer(in_ch, out_ch, kernel_size, stride)

        self.conv2 = ConvLayer(out_ch, out_ch, kernel_size,
                               stride, activation='linear')

    def forward(self, x):
        y = self.conv1(x)
        return self.conv2(y) + x


class DeconvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, activation='relu', upsample='nearest'):
        super(DeconvLayer, self).__init__()

        # upsample
        self.upsample = upsample

        # pad
        self.pad = nn.ReflectionPad2d(kernel_size//2)

        # conv
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)

        # activation
        if activation == 'prelu':
            self.activation = nn.PReLU()
        else:
            self.activation = lambda x: x

        # normalization
        self.normalization = channel.ChannelNorm2D_wrap(out_ch,
                                                        momentum=0.1, affine=True, track_running_stats=False)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample)
        x = self.pad(x)
        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x

# class ConvLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super(ConvLayer, self).__init__()
#         padding = kernel_size // 2
#         self.reflection_pad = nn.ReflectionPad2d(padding)
#         self.conv2d = nn.Conv2d(in_channels, out_channels,
#                                 kernel_size, stride)  # , padding)

#     def forward(self, x):
#         out = self.reflection_pad(x)
#         out = self.conv2d(out)
#         return out


# class UpsampleConvLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
#         super(UpsampleConvLayer, self).__init__()
#         self.upsample = upsample
#         if upsample:
#             # self.upsample = nn.Upsample(scale_factor=upsample, mode='bicubic', align_corners=True)
#             self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')

#         reflection_padding = kernel_size // 2
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
#         self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

#     def forward(self, x):
#         if self.upsample:
#             x = self.upsample(x)
#         out = self.reflection_pad(x)
#         out = self.conv2d(out)
#         return out


# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()

#         norm_kwargs = dict(momentum=0.1, affine=True,
#                            track_running_stats=False)

#         # channel.ChannelNorm2D_wrap
#         self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
#         self.in1 = channel.ChannelNorm2D_wrap(channels, **norm_kwargs)

#         self.relu = nn.ReLU()

#         self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
#         self.in2 = channel.ChannelNorm2D_wrap(channels, **norm_kwargs)

#     def forward(self, x):
#         identity = x
#         out = self.relu(self.in1(self.conv1(x)))
#         out = self.in2(self.conv2(out))

#         out += identity
#         out = self.relu(out)

#         return out


# class Generator(nn.Module):
#     def __init__(self, n_blocks = 6, in_feat = 64):
#         super(Generator, self).__init__()

#         # nonlineraity
#         self.relu = nn.PReLU()
#         self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)
#         self.tanh = nn.Tanh()

#         norm_kwargs = dict(momentum=0.1, affine=True,
#                            track_running_stats=False)

#         # self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
#         # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.unshuffle = PixelUnshuffle(2)

#         self.conv_1 = ConvLayer(12, 128, kernel_size=3, stride=1)

#         # self.conv_2 = ConvLayer(32, 64, kernel_size=3, stride=2)
#         # self.in2_e = channel.ChannelNorm2D_wrap(64, **norm_kwargs)

#         # self.conv_3 = ConvLayer(64, 128, kernel_size=3, stride=2)
#         # self.in3_e = channel.ChannelNorm2D_wrap(128, **norm_kwargs)

#         model = []
#         for i in range(n_blocks):
#             model += [ResidualBlock(128)]
#         self.resblocks = nn.Sequential(*model)

#         # self.res7 = ResidualBlock(128)
#         # self.res8 = ResidualBlock(128)
#         # self.res9 = ResidualBlock(128)

#         self.deconv_4 = UpsampleConvLayer(128, 128, kernel_size=3, stride=1)
#         self.in4_d = channel.ChannelNorm2D_wrap(128, **norm_kwargs)

#         self.deconv_3 = UpsampleConvLayer(
#             128, 64, kernel_size=3, stride=1, upsample=2)
#         self.in3_d = channel.ChannelNorm2D_wrap(64, **norm_kwargs)

#         # self.deconv_2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
#         # self.in2_d = channel.ChannelNorm2D_wrap(32, **norm_kwargs)

#         self.deconv_1 = UpsampleConvLayer(64, 3, kernel_size=3, stride=1)
#         # self.in1_d = channel.ChannelNorm2D_wrap(3, **norm_kwargs)

#     def forward(self, x):
#         # Recovery from encoder
#         # (3, 256, 268) -> (12, 128, 128)
#         y = self.unshuffle(x)

#         # (3, 256, 256) -> (3, 512, 512)
#         # y = self.upsample(y)

#         y = self.leakyRelu(self.conv_1(y))
#         # y = self.relu(self.in2_e(self.conv_2(y)))
#         # y = self.relu(self.in3_e(self.conv_3(y)))

#         residual = y
#         res = self.resblocks(y)

#         # res = self.res1(y)
#         # res = self.res2(res)
#         # res = self.res3(res)
#         # res = self.res4(res)
#         # res = self.res5(res)
#         # res = self.res6(res)
#         # res = self.res7(res)
#         # res = self.res8(res)
#         # res = self.res9(res)

#         res = self.in4_d(self.deconv_4(res))

#         res = res + residual
#         y = self.leakyRelu(res)

#         y = self.relu(self.in3_d(self.deconv_3(y)))
#         # y = self.relu(self.in2_d(self.deconv_2(y)))
#         out = self.tanh(self.deconv_1(y))
#         # y = self.conv_1(y)
#         # y = self.conv_2(y)
#         # y = self.conv_3(y)

#         # residual = y
#         # res = None
#         # for m in range(self.n_residual_blocks):
#         #     resblock_m = getattr(self, f'resblock_{str(m)}')
#         #     if m == 0:
#         #         res = resblock_m(y)
#         #     else:
#         #         res = resblock_m(res)

#         # res = self.deconv_4(res)
#         # res = res + residual
#         # y = self.leakyRelu(res)

#         # y = self.deconv_3(y)
#         # y = self.deconv_2(y)
#         # out = self.deconv_1(y)

#         return out


def trial():
    x = torch.randn((2, 3, 256, 256))

    G = Generator()
    x_generated = G(x)

    print(x.size())
    print(x_generated.size())

    save_img(x[0], 'interm/test.png')
    save_img(x_generated[0].detach(), 'interm/test_gen.png')
