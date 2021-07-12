import torch
import torch.nn as nn
import torchinfo
import torch.optim as optim

from utils import ConvLayer, DeconvLayer, FeatureExtractor


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, norm='skip', activation='skip'):
        super(ConvLayer, self).__init__()
        assert(norm in ('skip', 'channel'))
        assert(activation in ('skip', 'relu', 'prelu'))

        self.pad = nn.ZeroPad2d(kernel//2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        self.norm = None
        self.activation = None

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)

        return x


class PixCNN(nn.Module):
    def __init__(self, n_features=64):
        super(PixCNN, self).__init__()

        self.conv_blocks = nn.Sequential(
            ConvLayer(3, n_features, 3, 1, activation='relu'),
            ConvLayer(n_features, n_features, 3, 1, activation='relu'),
            ConvLayer(n_features, 3, 3, 1),
        )

    def forward(self, x):
        out = self.conv_blocks(x)

        out += x
        return out


class ResidualBlockNext(nn.Module):
    def __init__(self, num_layers=5, n_features=64):
        super(ResidualBlockNext, self).__init__()

        self.conv_1 = ConvLayer(3, n_features, 3, 1)
        self.conv_2 = ConvLayer(n_features, n_features, 3, 1)
        self.conv_3 = ConvLayer(n_features, 3, 7, 1)

        blocks = []
        blocks.append(ConvLayer(n_features, n_features*2, 7,
                      1, activation='relu'))
        for _ in range(num_layers-2):
            blocks.append(ConvLayer(n_features*2, n_features*2, 7,
                          1, activation='relu'))
        blocks.append(ConvLayer(n_features*2, n_features, 3, 1))

        self.conv_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out_conv1 = self.conv_1(x)
        out_blocks = self.conv_blocks(out_conv1)

        out_blocks += out_conv1
        out_conv2 = self.conv_2(out_blocks)

        out_conv2 += out_conv1
        out_conv3 = self.conv_3(out_conv2)

        return out_conv3


class ResidualBlock(nn.Module):
    def __init__(self, num_layers=5, n_features=64):
        super(ResidualBlock, self).__init__()
        assert(num_layers - 2 > 0)

        blocks = []
        blocks.append(ConvLayer(3, n_features, 7, 1, activation='relu'))
        for _ in range(num_layers-2):
            blocks.append(ConvLayer(n_features, n_features,
                          7, 1, activation='relu'))
        blocks.append(ConvLayer(n_features, 3, 7, 1))

        self.conv_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.conv_blocks(x)

        out += x
        return out


class ModifiedResidualModel(nn.Module):
    def __init__(self):
        super(ModifiedResidualModel, self).__init__()

        self.residual_convolutions_first = ResidualBlock()
        self.residual_convolutions_end = ResidualBlockNext()

    def forward(self, x):
        x = self.residual_convolutions_first(x)
        out = self.residual_convolutions_end(x)

        return out


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down1 = ConvLayer(3, 32, norm=False)
        self.down2 = ConvLayer(32, 64)
        self.down3 = ConvLayer(64, 128)
        self.down4 = ConvLayer(128, 256)
        self.down5 = ConvLayer(256, 256)
        self.down6 = ConvLayer(256, 256)
        self.down7 = ConvLayer(256, 256)
        self.down8 = ConvLayer(256, 256)  # 256x2x2 -> 1x1

        self.up7 = DeconvLayer(256, 256)  # 512x2x2
        self.up6 = DeconvLayer(2 * 256, 256)
        self.up5 = DeconvLayer(2 * 256, 256)
        self.up4 = DeconvLayer(2 * 256, 256)
        self.up3 = DeconvLayer(2 * 256, 128)
        self.up2 = DeconvLayer(2 * 128, 64)
        self.up1 = DeconvLayer(2 * 64, 32)

        self.out = nn.ConvTranspose2d(64, 3, 4, 2, padding=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        x = self.up7(d8)
        x = torch.cat([x, d7], dim=1)
        x = self.up6(x)
        x = torch.cat([x, d6], dim=1)
        x = self.up5(x)
        x = torch.cat([x, d5], dim=1)
        x = self.up4(x)
        x = torch.cat([x, d4], dim=1)
        x = self.up3(x)
        x = torch.cat([x, d3], dim=1)
        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.up1(x)
        x = torch.cat([x, d1], dim=1)

        out = self.out(x)
        return out


class Model(nn.Module):
    def __init__(self, device, model, opt={}):
        super(Model, self).__init__()
        assert(model in ('unet', 'mod_resblocks', 'pixcnn'))
        assert(opt.criterion in ('mse', 'vgg19'))

        decay = 0.0
        lr = opt.lr
        self.criterion = opt.criterion

        self.use_gradient_clipping = False

        if self.use_gradient_clipping:
            print('using gradient clipping')

        if model == 'unet':
            self.model = UNet().to(device)
        elif model == 'mod_resblocks':
            decay = 0.0001
            self.model = ModifiedResidualModel().to(device)
        elif model == 'pixcnn':
            self.model = PixCNN().to(device)

        if model == 'mod_resblocks_w0':
            print('using SGD')

            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9, weight_decay=decay)
        else:
            print('using Adam')

            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=decay)

        self.mse = nn.MSELoss()
        self.vgg19 = FeatureExtractor().to(device)
        self.scheduler = None

        if model == 'mod_resblocks':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=4, gamma=0.5)

        self.input = None
        self.ground_truth = None
        self.output = None

    def vgg19_loss(self, output, target):
        output = self.vgg19(output)
        target = self.vgg19(target)
        return self.mse(output, target)

    def step_scheduler(self):
        if self.scheduler is not None:
            lr_before = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            if lr_before != lr:
                print('Learning rate updated to: %.7f' % lr)

    def forward(self, x):
        return self.model(x)

    def set_requires_grad_cs(self, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(self.model, list):
            nets = [self.model]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, input, ground_truth):
        self.input = input
        self.ground_truth = ground_truth

    def get_losses(self):
        if self.criterion == 'mse':
            return self.mse(self.output, self.ground_truth)
        elif self.criterion == 'vgg19':
            return self.vgg19_loss(self.output, self.ground_truth)

    def optimize(self):
        self.optimizer.zero_grad()
        self.output = self.forward(self.input)
        loss = self.get_losses()
        loss.backward()
        if self.use_gradient_clipping:
            nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
        self.optimizer.step()

        return loss.item()


if __name__ == '__main__':
    net = Model(torch.device('cpu'), 'mod_resblocks')

    torchinfo.summary(net, input_size=(1, 3, 256, 256))

    # model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)
