import cv2
import numpy as np
from skimage.metrics import structural_similarity


def ssim(image_out, image_ref):
    image_out = np.array(image_out, dtype='float')
    image_ref = np.array(image_ref, dtype='float')

    return structural_similarity(image_out, image_ref, multichannel=True, data_range=255.)


def psnr(ground, compressed):
    np_ground = np.array(ground, dtype='float')
    np_compressed = np.array(compressed, dtype='float')
    mse = np.mean((np_ground - np_compressed)**2)
    psnr = np.log10(255**2/mse) * 10
    return psnr
