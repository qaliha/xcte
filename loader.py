from os import listdir
from os.path import join, exists

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from generate_dataset import dir_exists, mkdir

from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".PNG", ".jpg", ".jpeg"])

transform_list = [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform_list_compose = transforms.Compose(transform_list)
def transform_and_normalize(img):
    return transform_list_compose(img)

def normalize(img):
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        if not dir_exists(self.b_path):
            mkdir(self.b_path)

    def __getitem__(self, index):
        self.current_index = index

        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        a = transform_and_normalize(a)

        return a

    def __len__(self):
        return len(self.image_filenames)
