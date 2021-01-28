
import torch
import torch.nn as nn


class DiscriminatorHF(nn.Module):
    def __init__(self, spectral_norm=True):
        """ 
        Convolutional patchGAN discriminator used in [1].
        Accepts as input generator output G(z) or x ~ p*(x) where
        p*(x) is the true data distribution.
        Contextual information provided is encoder output y = E(x)
        ========
        Arguments:
        image_dims:     Dimensions of input image, (C_in,H,W)
        context_dims:   Dimensions of contextual information, (C_in', H', W')
        C:              Bottleneck depth, controls bits-per-pixel
                        C = 220 used in [1], C = C_in' if encoder output used
                        as context.

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        super(DiscriminatorHF, self).__init__()

        kernel_dim = 4
        filters = (64, 128, 256, 512)

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=1, padding_mode='reflect')
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        if spectral_norm is True:
            norm = nn.utils.spectral_norm
        else:
            norm = nn.utils.weight_norm

        # (3, 256,256) -> (64,128,128), with implicit padding
        self.conv1 = norm(nn.Conv2d(6, filters[0], kernel_dim, **cnn_kwargs))

        # (128,128) -> (64,64)
        self.conv2 = norm(
            nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs))

        # (64,64) -> (32,32)
        self.conv3 = norm(
            nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs))

        # (32,32) -> (16,16)
        self.conv4 = norm(
            nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs))

        self.conv_out = nn.Conv2d(filters[3], 1, kernel_size=1, stride=1)

    def forward(self, x):
        """
        x: Concatenated real/gen images
        """
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        out_logits = self.conv_out(x).view(-1, 1)
        out = torch.sigmoid(out_logits)

        return out, out_logits


def trial():
    x = torch.randn((2, 3, 256, 256))
    y = torch.randn((2, 3, 256, 256))

    D = DiscriminatorHF()
    out, out_logits = D(torch.cat((x, y), dim=1))

    # print(out.size(), out_logits.size())
    print(out)
    print(out_logits)
