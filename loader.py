from os import listdir
from os.path import join

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".PNG", ".jpg", ".jpeg"])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        # self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        # b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        
        a = transforms.ToTensor()(a)
        # b = transforms.ToTensor()(b)
    
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        # b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        return a
        # if self.direction == "a2b":
        #     return a, b
        # else:
        #     return b, a

    def __len__(self):
        return len(self.image_filenames)
