import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.norm import channel
from src.utils.compression import _compress
from src.networks.generator import ConvLayer


class FeatureExtractor(nn.Module):
    def __init__(self, n_features):
        super(FeatureExtractor, self).__init__()

        self.conv_block_1 = ConvLayer(3, n_features, 5, 1)
        self.conv_block_2 = ConvLayer(n_features, n_features, 3, 1)
        self.conv_block_3 = ConvLayer(
            n_features, 3, 3, 1, activation='skip', norm='skip')

    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)

        return out


class Encoder(nn.Module):
    def __init__(self, cuda=False, alpha=.8, n_features=64):
        super(Encoder, self).__init__()

        self.feature_net = FeatureExtractor(n_features)

        self.connection_weights = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        inp = x
        out = self.feature_net(x)

        out += inp

        return out


def trial():
    x = torch.randn((2, 3, 256, 256))

    E = Encoder()
    x_encoded = E(x)
    compressed_encoded = _compress(x_encoded, 2)

    print(compressed_encoded.size())
    print(x.size())
    print(x_encoded.size())
