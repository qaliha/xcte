import torch
import torch.nn as nn
import torchinfo
import torch.optim as optim

from .utils import ConvLayer, DeconvLayer, FeatureExtractor


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
        assert(model in ('unet'))
        assert(opt.criterion in ('mse', 'vgg19'))

        if model == 'unet':
            self.model = UNet().to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        self.criterion = opt.criterion
        self.mse = nn.MSELoss()
        self.vgg19 = FeatureExtractor().to(device)

        self.input = None
        self.ground_truth = None
        self.output = None

    def vgg19_loss(self, output, target):
        output = self.vgg19(output)
        target = self.vgg19(target)
        return self.mse(output, target)

    def forward(self, x):
        return self.model(x)

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
        self.optimizer.step()

        return loss.item()


if __name__ == '__main__':
    net = Model(model='unet')

    torchinfo.summary(net, input_size=(1, 3, 256, 256))

    # model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)
