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

        self.conv_block_1 = ConvLayer(
            3, n_features, 3, 1, norm='none', activation='prelu')
        self.conv_block_2 = ConvLayer(
            n_features, n_features, 3, 1, norm='none', activation='prelu')
        # Ok for now remove this and copy the reference networks
        # model += [ConvLayer(n_features, n_features, 3, 1, norm='skip')]
        self.conv_block_downsample = ConvLayer(
            n_features, 12, 2, 2, norm='none', activation='none', padding='none')

        # Is padding required here and using kernel 3? maybe not neccesarry because this network not as deep as generator and it's last layer
        # Padding: padding='reflection', reflection_padding=(0, 1, 1, 0)

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

        initialized_tensor = torch.tensor(
            data=[alpha, alpha, alpha], dtype=torch.float32)
        initialized_tensor = initialized_tensor.unsqueeze(dim=0)
        initialized_tensor = initialized_tensor.unsqueeze(dim=2)
        initialized_tensor = initialized_tensor.unsqueeze(dim=3)

        self.connection_weights = nn.Parameter(initialized_tensor)

    def forward(self, x):
        inp = x
        # Get or extract the feature
        y = self.feature_net(x)
        out = F.normalize(y, p=2, dim=1)

        out = self.connection_weights * inp + \
            (1 - self.connection_weights) * out

        return out


def trial():
    x = torch.randn((2, 3, 256, 256))

    E = Encoder()
    x_encoded = E(x)
    compressed_encoded = _compress(x_encoded, 2)

    print(compressed_encoded.size())
    print(x.size())
    print(x_encoded.size())
