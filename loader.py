import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".PNG", ".jpg", ".jpeg"])


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()

        self.a_path = os.path.join(image_dir, "a")
        self.b_path = os.path.join(image_dir, "b")
        self.image_filenames = [x for x in os.listdir(
            self.a_path) if is_image_file(x)]

    def _transform(self):
        transforms_list = [
            transforms.ToTensor()
        ]

        return transforms.Compose(transforms_list)

    def random(self):
        return torch.rand(1).numpy()[0]

    def __getitem__(self, index):
        a = Image.open(os.path.join(
            self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(os.path.join(
            self.b_path, self.image_filenames[index])).convert('RGB')

        default_transforms = self._transform()

        a_tensor = default_transforms(a)
        b_tensor = default_transforms(b)

        return a_tensor, b_tensor

    def __len__(self):
        return len(self.image_filenames)
