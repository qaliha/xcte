import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding='default', activation='relu', norm='channel', reflection_padding=3, cnn_kwargs=dict()):
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
        elif activation == 'relu6':
            self.activation = nn.ReLU6()
        elif activation == 'rrelu':
            self.activation = nn.RReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.activation = None

        # normalization
        if norm == 'group':
            self.normalization = nn.GroupNorm(4, out_ch)
        elif norm == 'batch':
            self.normalization = nn.BatchNorm2d(out_ch)
        elif norm == 'instance':
            self.normalization = nn.InstanceNorm2d(
                out_ch, momentum=0.1, affine=True, track_running_stats=False)
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
