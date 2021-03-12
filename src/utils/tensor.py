import numpy as np
from torchvision import transforms

from PIL import Image


def tensor2img(tensor):
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    tensor = np.squeeze(tensor)
    tensor = np.moveaxis(tensor, 0, 2)
    tensor = tensor * 255
    tensor = tensor.clip(0, 255).astype(np.uint8)

    img = Image.fromarray(tensor)
    return img

# Good for [0, 1] image range aka save compressed image


def save_img(image_tensor, filename):
    image = tensor2img(image_tensor)

    image.save(filename)


def normalize_input_from_normalied(image_tensor):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)

    image = Image.fromarray(image_numpy)
    return image

# Change from [-1. 1] to [0, 255]


def prepare_for_compression_from_normalized_input(image_tensor):
    image = normalize_input_from_normalied(image_tensor)

    return transforms.ToTensor()(image)

# Good for save [-1, 1] tensor image


def save_img_version(image_tensor, filename):
    # Because we disable the normalization
    save_img(image_tensor, filename)
    # image_pil = normalize_input_from_normalied(image_tensor)

    # image_pil.save(filename)

# def save_img(image_tensor, filename):
#     image_numpy = image_tensor.float().numpy()
#     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
#     image_numpy = image_numpy.clip(0, 255)
#     image_numpy = image_numpy.astype(np.uint8)
#     image_pil = Image.fromarray(image_numpy)
#     image_pil.save(filename)
