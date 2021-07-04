import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from os.path import join
from os import listdir
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Here we will use the following layers and make an array of their indices
        # 0: block1_conv1
        # 5: block2_conv1
        # 10: block3_conv1
        # 19: block4_conv1
        # 28: block5_conv1
        self.req_features = ['0']
        self.model = models.vgg19(
            pretrained=True).features[:int(self.req_features[-1])].eval()

        # freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False

    # x holds the input tensor(image) that will be feeded to each layer
    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if (str(layer_num) in self.req_features):
                features.append(x)

        return features[0]


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True):
        super(ConvLayer, self).__init__()
        blocks = []
        blocks.append(nn.Conv2d(in_channels, out_channels,
                      kernel_size=4, stride=2, padding=1))
        if norm:
            blocks.append(nn.BatchNorm2d(out_channels))
        blocks.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True):
        super(DeconvLayer, self).__init__()
        blocks = []
        blocks.append(nn.ConvTranspose2d(in_channels, out_channels,
                      kernel_size=4, stride=2, padding=1))
        if norm:
            blocks.append(nn.BatchNorm2d(out_channels))
        blocks.append(nn.ReLU())
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".PNG", ".jpg", ".jpeg"])


def mkdir(directory, mode=0o777):
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.chmod(directory, mode=mode)


def dir_exists(directory):
    return os.path.exists(directory)


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(
            self.a_path) if is_image_file(x)]

    def _transforms(self):
        # Default one, to tensor and normalize
        transforms_list = [
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        return transforms.Compose(transforms_list)

    def __getitem__(self, index):
        a = Image.open(
            join(self.a_path, self.image_filenames[index])).convert('RGB')

        b = Image.open(
            join(self.b_path, self.image_filenames[index])).convert('RGB')

        default_transforms = self._transforms()

        a_tensor = default_transforms(a)
        b_tensor = default_transforms(b)

        return a_tensor, b_tensor

    def __len__(self):
        return len(self.image_filenames)


def get_training_set(root_dir):
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir)


def get_test_set(root_dir):
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir)


def tensor2img(tensor):
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    tensor = np.squeeze(tensor)
    tensor = np.moveaxis(tensor, 0, 2)
    tensor = (tensor * 255)  # + 0.5  # ? add 0.5 to rounding
    tensor = tensor.clip(0, 255).astype(np.uint8)

    img = Image.fromarray(tensor)
    return img


def psnr(ground, compressed):
    np_ground = np.array(ground, dtype='float')
    np_compressed = np.array(compressed, dtype='float')
    mse = np.mean((np_ground - np_compressed)**2)
    psnr = np.log10(255**2/mse) * 10
    return psnr
