import os
import torch
import torchvision
import numpy as np

from sklearn.feature_extraction import image
from PIL import Image

# Load 100 images
# Random crop the images to 16x16
# Make grid 10x10

image_train_folders = "D:\\Bismillah\\xcte\\dataset_test\\DIV2K_train_LR_bicubic\\X4"
image_valid_folders = "D:\\Bismillah\\xcte\\dataset_test\\DIV2K_valid_LR_bicubic\\X4"
image_test_folders = "D:\\Bismillah\\xcte\\dataset_test\\testing"

# Generate train visualizer

max_patches = 100
patch_size = (64, 64)


def tensor2img(tensor):
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    tensor = np.squeeze(tensor)
    tensor = np.moveaxis(tensor, 0, 2)
    tensor = (tensor * 255)  # + 0.5  # ? add 0.5 to rounding
    tensor = tensor.clip(0, 255).astype(np.uint8)

    img = Image.fromarray(tensor)
    return img


def make_visualizer(folder_path, name):
    img_files = os.listdir(folder_path)

    grid_tensors = list()

    for file in img_files[0:max_patches]:
        image_pil = Image.open(os.path.join(folder_path, file)).convert('RGB')
        image_pil = np.array(image_pil)

        images = image.extract_patches_2d(
            image_pil, patch_size, max_patches=2, random_state=0)

        if len(images) > 0:
            selected_image = images[0]
            grid_tensors.append(torchvision.transforms.ToTensor()(
                Image.fromarray(selected_image)))

    tensor_grid = torchvision.utils.make_grid(
        grid_tensors, nrow=10, pad_value=255)

    torchvision.utils.save_image(tensor_grid, 'dataset_test/' + name + '.png')


make_visualizer(image_train_folders, 'training_data')
make_visualizer(image_valid_folders, 'validaiton_data')


def make_visualizerv2(folder_path, name):
    img_files = os.listdir(folder_path)

    grid_tensors = list()

    j = 0
    for file in img_files[0:max_patches]:
        image_pil = Image.open(os.path.join(folder_path, file)).convert('RGB')
        image_pil = np.array(image_pil)

        images = image.extract_patches_2d(
            image_pil, patch_size, max_patches=5, random_state=0)

        if len(images) > 0:
            for i in range(5):
                if j >= 100:
                    break

                selected_image = images[i]
                grid_tensors.append(torchvision.transforms.ToTensor()(
                    Image.fromarray(selected_image)))

                j += 1

    tensor_grid = torchvision.utils.make_grid(
        grid_tensors, nrow=10, pad_value=255)

    torchvision.utils.save_image(tensor_grid, 'dataset_test/' + name + '.png')


make_visualizerv2(image_test_folders, 'testing_data')
