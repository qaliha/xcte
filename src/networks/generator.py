from src.utils.tensor import save_img
from numpy.core.numeric import identity
import torch
import torch.nn as nn

from src.norm import channel
from src.utils.pixelunshuffle import PixelUnshuffle


class Generator(nn.Module):
    def __init__(self, n_blocks=6, n_feature=64):
        super(Generator, self).__init__()

        self.n_blocks = n_blocks

        norm_kwargs = dict(momentum=0.1, affine=True,
                           track_running_stats=False)

        self.pre_normalization = channel.ChannelNorm2D_wrap(3, **norm_kwargs)
        self.unshuffle = PixelUnshuffle(2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.pre_normalization = nn.BatchNorm2d(12)

        self.conv_block_1 = ConvLayer(12, n_feature, 3, 1)
        self.conv_block_before_resblock = ConvLayer(
            n_feature, n_feature, 3, 1, activation='leaky')

        for m in range(n_blocks):
            resblock_m = ResidualLayer(n_feature, n_feature, 3, 1)
            self.add_module(f'resblock_{str(m)}', resblock_m)

        self.conv_block_after_resblock = ConvLayer(
            n_feature, n_feature, 3, 1, activation='leaky')

        self.conv_block_2 = ConvLayer(n_feature, n_feature, 3, 1)
        self.conv_block_out = ConvLayer(
            n_feature, 3, 3, 1, norm='none', activation='none')

    def forward(self, x):
        head = self.unshuffle(x)
        head = self.upsampling(head)
        # head = self.pre_normalization(head)
        head = self.conv_block_1(head)
        head = self.conv_block_before_resblock(head)

        for m in range(self.n_blocks):
            resblock_m = getattr(self, f'resblock_{str(m)}')
            if m == 0:
                x = resblock_m(head)
            else:
                x = resblock_m(x)

        x += head
        x = self.conv_block_after_resblock(x)
        x = self.conv_block_2(x)
        out = self.conv_block_out(x)

        return out


# class ConvTransposeLayer(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size, stride, cnn_kwargs=dict()):
#         super(ConvTransposeLayer, self).__init__()

#         # convolution
#         self.conv_layer = nn.ConvTranspose2d(
#             in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1, **cnn_kwargs)

#         self.pre_norm = channel.ChannelNorm2D_wrap(
#             in_ch, momentum=0.1, affine=True, track_running_stats=False)
#         self.normalization = channel.ChannelNorm2D_wrap(
#             out_ch, momentum=0.1, affine=True, track_running_stats=False)

#     def forward(self, x):
#         # normalize the input
#         out = self.pre_norm(x)
#         out = self.conv_layer(out)
#         out = self.normalization(out)

#         return out


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding='default', activation='prelu', norm='channel', reflection_padding=3, cnn_kwargs=dict()):
        super(ConvLayer, self).__init__()

        # padding
        if padding == 'default':
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        elif padding == 'reflection':
            self.pad = nn.ReflectionPad2d(reflection_padding)
        elif padding == 'zero':
            self.pad = nn.ZeroPad2d(kernel_size//2)
        else:
            self.pad = None

        # convolution
        self.conv_layer = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, **cnn_kwargs)

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
        elif norm == 'group':
            self.normalization = nn.GroupNorm(4, out_ch)
        elif norm == 'batch':
            self.normalization = nn.BatchNorm2d(out_ch)
        else:
            self.normalization = None

    def forward(self, x):
        if self.pad is not None:
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

        self.conv1 = ConvLayer(in_ch, out_ch, kernel_size,
                               stride)

        self.conv2 = ConvLayer(out_ch, out_ch, kernel_size,
                               stride, activation='none')

    def forward(self, x):
        identity_map = x
        res = self.conv1(x)
        res = self.conv2(res)

        return torch.add(res, identity_map)


# class DeconvLayer(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size, stride, activation='relu', upsample='nearest', norm='batch'):
#         super(DeconvLayer, self).__init__()

#         # upsample
#         self.upsample = nn.Upsample(scale_factor=2, mode=upsample)

#         # pad
#         self.pad = nn.ReflectionPad2d(kernel_size//2)

#         # conv
#         self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)

#         # activation
#         if activation == 'prelu':
#             self.activation = nn.PReLU()
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         else:
#             raise NotImplementedError("Not implemented!")

#         # normalization
#         if norm == 'channel':
#             self.normalization = channel.InstanceNorm2D_wrap(out_ch,
#                                                              momentum=0.1, affine=True, track_running_stats=False)
#         elif norm == 'group':
#             self.normalization = nn.GroupNorm(2, out_ch, affine=True)
#         elif norm == 'batch':
#             self.normalization = nn.BatchNorm2d(out_ch)
#         else:
#             self.normalization = None

#     def forward(self, x):
#         x = self.upsample(x)
#         x = self.pad(x)
#         x = self.conv(x)
#         if self.normalization is not None:
#             x = self.normalization(x)
#         x = self.activation(x)
#         return x


def trial():
    x = torch.randn((2, 3, 256, 256))

    G = Generator()
    x_generated = G(x)

    print(x.size())
    print(x_generated.size())

    save_img(x[0], 'interm/test.png')
    save_img(x_generated[0].detach(), 'interm/test_gen.png')
