import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from torch.cuda.amp import custom_bwd, custom_fwd


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)


def tensor2img(tensor):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(
        1, 2, 0).to('cpu', torch.uint8).numpy()

    img = Image.fromarray(ndarr)
    return img

# def tensor2img(tensor):
#     tensor = tensor.cpu()
#     tensor = tensor.detach().numpy()
#     tensor = np.squeeze(tensor)
#     tensor = np.moveaxis(tensor, 0, 2)
#     tensor = tensor * 255
#     tensor = tensor.clip(0, 255).astype(np.uint8)

#     img = Image.fromarray(tensor)
#     return img

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
