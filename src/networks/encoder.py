import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from src.norm import channel
from src.utils.compression import compress

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.activation = nn.LeakyReLU(0.2)
        norm = channel.ChannelNorm2D_wrap

        cnn_kwargs = dict(stride=2, padding=0, padding_mode='reflect')
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)

        self.pre_pad = nn.ReflectionPad2d(3)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(1)

        # (256,256) -> (256,256), with implicit padding
        self.conv_block1 = nn.Sequential(
            self.pre_pad,
            nn.Conv2d(3, 8, kernel_size=(7,7), stride=1),
            self.activation,
        )

        # (256,256) -> (128,128)
        self.conv_block2 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(8, 16, 3, **cnn_kwargs),
            norm(16, **norm_kwargs),
            self.activation,
        )

        # (128,128) -> (64,64)
        self.conv_block3 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(16, 32, 3, **cnn_kwargs),
            norm(32, **norm_kwargs),
            self.activation,
        )

        # (64,64) -> (32,32)
        self.conv_block4 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(32, 64, 3, **cnn_kwargs),
            norm(64, **norm_kwargs),
            self.activation,
        )
        
        # Project channels onto space w/ dimension C
        # Feature maps have dimension C x W/16 x H/16
        # (32,32) -> (32,32)
        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(64, 3, 3, stride=1),
            norm(3, **norm_kwargs),
            self.activation,
        )

        # Upsample
        self.context_conv = nn.Conv2d(3, 12, kernel_size=3, padding=1, padding_mode='reflect')

        # (*, 12, 128, 128) -> (*, 3, 256, 256)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.context_upsample = nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, x):
        y = self.conv_block1(x)
        y = self.conv_block2(y)
        y = self.conv_block3(y)
        y = self.conv_block4(y)
        y = self.conv_block_out(y)

        y = self.activation(self.context_conv(y))

        # Upsample to get features
        y = self.context_upsample(y)
        y = self.pixel_shuffle(y)

        return y

class Encoder(nn.Module):
    def __init__(self, cuda=False):
        super(Encoder, self).__init__()

        self.feature_net = FeatureExtractor()
        
        self.connection_weights = torch.empty(3, 256, 256).uniform_(0, 1)
        if cuda:
            self.connection_weights = self.connection_weights.to(torch.device("cuda:0"))
            # self.connection_weights.requires_grad_(True)

    def forward(self, x):
        inp = x
        # Get or extract the feature
        y = self.feature_net(x)

        out = F.normalize(y, p=2, dim=1)

        out = torch.lerp(out, inp, self.connection_weights)
        # out = self.connection_weights * inp + (1 - self.connection_weights) * out

        return out

def trial():
    x = torch.randn((2, 3, 256, 256))

    E = Encoder()
    x_encoded = E(x)
    compressed_encoded = compress(x_encoded, 2)

    print(compressed_encoded.size())
    print(x.size())
    print(x_encoded.size())