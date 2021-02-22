from os import listdir
import math
from os.path import join, exists
import random
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from generate_dataset import dir_exists, mkdir

from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".PNG", ".jpg", ".jpeg"])

# transform_list = [transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

# transform_list_compose = transforms.Compose(transform_list)
# def transform_and_normalize(img):
#     return transform_list_compose(img)

# def normalize(img):
#     return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)


SCALE_MIN = 1.05
SCALE_MAX = 1.25


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, scale_n_crop):
        super(DatasetFromFolder, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(
            self.a_path) if is_image_file(x)]

        self.scale_n_crop = scale_n_crop

        if not dir_exists(self.b_path):
            mkdir(self.b_path)

    def _transforms(self):
        # Default one, to tensor and normalize
        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        return transforms.Compose(transforms_list)

    def __getitem__(self, index):
        a = Image.open(
            join(self.a_path, self.image_filenames[index])).convert('RGB')

        # Current image size
        image_size = a.size[0]

        default_transforms = self._transforms()

        a_tensor = default_transforms(a)
        a_composed = list()

        b = list()
        b_tensor = list()
        b_composed = list()
        b_path = join(self.b_path, self.image_filenames[index])

        b_found = exists(b_path)

        if exists(b_path):
            b = Image.open(b_path).convert('RGB')

            b_tensor = default_transforms(b)

        # Randomly apply augmentation to dataset
        # Random horizontal flipping
        if random.random() > 0.5:
            a_composed = TF.hflip(a)
            if b_found:
                b_composed = TF.hflip(b)
        else:
            a_composed = a
            if b_found:
                b_composed = b

        # Random vertical flipping
        if random.random() > 0.5:
            a_composed = TF.vflip(a_composed)
            if b_found:
                b_composed = TF.vflip(b_composed)

        # Do scaling and randomly crop the image if enabled
        if self.scale_n_crop:
            scale = np.random.uniform(SCALE_MIN, SCALE_MAX)

            resize = transforms.Resize(
                (math.ceil(scale * image_size), math.ceil(scale * image_size)))

            a_composed = resize(a_composed)
            if b_found:
                b_composed = resize(b_composed)

            # Randomly crop the image
            i, j, h, w = transforms.RandomCrop.get_params(
                a_composed, output_size=(image_size, image_size))
            a_composed = TF.crop(a_composed, i, j, h, w)
            if b_found:
                b_composed = TF.crop(b_composed, i, j, h, w)

        a_composed = default_transforms(a_composed)
        if b_found:
            b_composed = default_transforms(b_composed)

        return a_tensor, b_tensor, b_path, a_composed, b_composed

    # def __getitem__(self, index):
    #     a = Image.open(
    #         join(self.a_path, self.image_filenames[index])).convert('RGB')

    #     image_size = a.size[0]
    #     image_bicubic = (image_size + 30) if self.scale_n_crop else image_size

    #     # Crop offset
    #     w_offset = random.randint(0, max(0, image_bicubic - image_size - 1))
    #     h_offset = random.randint(0, max(0, image_bicubic - image_size - 1))

    #     a_resized = a.resize((image_bicubic, image_bicubic), Image.NEAREST)

    #     a = transforms.ToTensor()(a)
    #     a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)

    #     a_resized = transforms.ToTensor()(a_resized)

    #     a_resized = a_resized[:, h_offset:h_offset +
    #                           image_size, w_offset:w_offset + image_size]

    #     a_resized = transforms.Normalize(
    #         (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a_resized)

    #     b = list()
    #     b_resized = list()
    #     b_path = join(self.b_path, self.image_filenames[index])

    #     if exists(b_path):
    #         b = Image.open(b_path).convert('RGB')

    #         b_resized = b.resize((image_bicubic, image_bicubic), Image.BICUBIC)

    #         b = transforms.ToTensor()(b)
    #         b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

    #         b_resized = transforms.ToTensor()(b_resized)

    #         b_resized = b_resized[:, h_offset:h_offset +
    #                               image_size, w_offset:w_offset + image_size]

    #         b_resized = transforms.Normalize(
    #             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b_resized)

    #     return a, b, b_path, a_resized, b_resized

    def __len__(self):
        return len(self.image_filenames)
