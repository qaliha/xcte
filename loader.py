from os import listdir
from os.path import join, exists
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

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


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(
            self.a_path) if is_image_file(x)]

        if not dir_exists(self.b_path):
            mkdir(self.b_path)

    def __getitem__(self, index):
        # Crop offset
        w_offset = random.randint(0, max(0, 286 - 256 - 1))
        h_offset = random.randint(0, max(0, 286 - 256 - 1))

        a = Image.open(
            join(self.a_path, self.image_filenames[index])).convert('RGB')

        a_resized = a.resize((286, 286), Image.BICUBIC)

        a = transforms.ToTensor()(a)
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)

        a_resized = transforms.ToTensor()(a_resized)

        a_resized = a_resized[:, h_offset:h_offset +
                              256, w_offset:w_offset + 256]

        a_resized = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a_resized)

        b = list()
        b_resized = list()
        b_path = join(self.b_path, self.image_filenames[index])

        if exists(b_path):
            b = Image.open(b_path).convert('RGB')

            b_resized = b.resize((286, 286), Image.BICUBIC)

            b = transforms.ToTensor()(b)
            b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

            b_resized = transforms.ToTensor()(b_resized)

            b_resized = b_resized[:, h_offset:h_offset +
                                  256, w_offset:w_offset + 256]

            b_resized = transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b_resized)

        return a, b, b_path, a_resized, b_resized

    def __len__(self):
        return len(self.image_filenames)
