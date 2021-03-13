import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.norm import channel
from src.utils.compression import _compress
from src.networks.generator import ConvLayer


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        n_features = 64

        self.conv_block_1 = ConvLayer(3, n_features, 3, 1, norm='none')
        self.conv_block_2 = ConvLayer(
            n_features, n_features, 3, 1, norm='none')
        # Ok for now remove this and copy the reference networks
        # model += [ConvLayer(n_features, n_features, 3, 1, norm='skip')]
        self.conv_block_downsample = ConvLayer(n_features, 12, 3, 2, norm='none',
                                               activation='none', padding='reflection', reflection_padding=(0, 1, 1, 0))

        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)

        downsampled = self.conv_block_downsample(out)
        shuffled = self.shuffle(downsampled)

        return shuffled


class Encoder(nn.Module):
    def __init__(self, cuda=False, alpha=.8):
        super(Encoder, self).__init__()

        self.feature_net = FeatureExtractor()

        # self.connection_weights = nn.Parameter(torch.empty(3, 256, 256).uniform_(0, 1))
        self.connection_weights = nn.Parameter(torch.tensor(alpha))
        # if cuda:
        #     self.connection_weights = self.connection_weights.to(torch.device("cuda:0"))
        # self.connection_weights.requires_grad_(True)

    def forward(self, x):
        inp = x
        # Get or extract the feature
        y = self.feature_net(x)
        out = F.normalize(y, p=2, dim=1)

        connection_restricted = self.connection_weights.sigmoid()

        out = connection_restricted * inp + \
            (1 - connection_restricted) * out

        return out


def trial():
    x = torch.randn((2, 3, 256, 256))

    E = Encoder()
    x_encoded = E(x)
    compressed_encoded = _compress(x_encoded, 2)

    print(compressed_encoded.size())
    print(x.size())
    print(x_encoded.size())
