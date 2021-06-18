import torch
import torch.nn as nn

from src.networks.conv import ConvLayer


class Model(nn.Module):
    def __init__(self, opt=None):
        super(Model, self).__init__()

        # hyper parameters, for now static
        d0 = 19
        num_feature = 64

        cnn_kwargs = dict(kernel_size=3, stride=1,
                          activation='relu', norm='skip', padding='zero')
        cnn_kwargs_output = dict(kernel_size=3, stride=1,
                                 activation='skip', norm='skip', padding='zero')

        gm_layers = list()
        drm_layers = list()

        gm_layers += [ConvLayer(3, num_feature, **cnn_kwargs)]
        for _ in range(d0-1):
            gm_layers += [ConvLayer(num_feature, num_feature, **cnn_kwargs)]
        gm_layers += [ConvLayer(num_feature, 3, **cnn_kwargs_output)]

        drm_layers += [ConvLayer(3, num_feature, **cnn_kwargs)]
        for _ in range(29-(d0+3)):
            drm_layers += [ConvLayer(num_feature, num_feature, **cnn_kwargs)]
        drm_layers += [ConvLayer(num_feature, 3, **cnn_kwargs_output)]

        self.gm_models = nn.Sequential(*gm_layers)
        self.drm_models = nn.Sequential(*drm_layers)

    def forward(self, x):
        # pass to GM
        y = self.gm_models(x)
        # substract input with output y
        out = x - y
        # pass to DRM
        y2 = self.drm_models(out)
        # Add substracted with new detail
        out2 = out + y2

        return out2
