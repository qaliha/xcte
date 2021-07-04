import argparse
import os
import numpy as np

from PIL import Image
from tqdm import tqdm
from sklearn.feature_extraction import image
from .utils import mkdir, dir_exists
import torch
import torchvision


def _compress(tensor, bit):
    max_val = 2**bit - 1
    tensor = torch.clamp(tensor, 0.0, 1.0) * max_val
    tensor = torch.round(tensor)
    tensor = tensor / max_val
    return tensor


def tensor2img(tensor):
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    tensor = np.squeeze(tensor)
    tensor = np.moveaxis(tensor, 0, 2)
    tensor = (tensor * 255)  # + 0.5  # ? add 0.5 to rounding
    tensor = tensor.clip(0, 255).astype(np.uint8)

    img = Image.fromarray(tensor)
    return img


def crop(img_arr, block_size):
    h_b, w_b = block_size
    v_splited = np.vsplit(img_arr, img_arr.shape[0]//h_b)
    h_splited = np.concatenate(
        [np.hsplit(col, img_arr.shape[1]//w_b) for col in v_splited], 0)
    return h_splited


def generate_patches(src_path, files, set_path, crop_size, img_format, max_patches, resize, local_j=0, max_n=0):

    local_local_j = local_j

    img_path = os.path.join(src_path, files)
    img = Image.open(img_path).convert('RGB')

    # jika resize factor, factor > 1 atau factor < -1
    if resize > 1 or resize < -1:
        wi, he = img.size
        if resize > 1:
            wi = wi * resize
            he = he * resize
        else:
            wi = wi // abs(resize)
            he = he // abs(resize)

        img = img.resize((wi, he), resample=Image.BICUBIC)

    name, _ = files.split('.')
    filedir = os.path.join(set_path, 'a')
    if not dir_exists(filedir):
        mkdir(filedir)

    filedirb = os.path.join(set_path, 'a')
    if not dir_exists(filedirb):
        mkdir(filedirb)

    img = np.array(img)
    h, w = img.shape[0], img.shape[1]

    if crop_size == None:
        img = np.copy(img)
        img_patches = np.expand_dims(img, 0)
    else:
        if resize > 1 or resize < -1:
            img_patches = image.extract_patches_2d(
                img, (crop_size[0], crop_size[1]), max_patches=max_patches, random_state=0)
        else:
            rem_h = (h % crop_size[0])
            rem_w = (w % crop_size[1])
            img = img[:h-rem_h, :w-rem_w]
            img_patches = crop(img, crop_size)

    n = 0

    for i in range(min(len(img_patches), max_patches)):
        img = Image.fromarray(img_patches[i])

        tensor = torchvision.transforms.ToTensor()(img)
        compressed_image = _compress(tensor, 3)

        img_compressed = tensor2img(compressed_image)

        img.save(
            os.path.join(filedir, '{}_{}.{}'.format(name, i, img_format))
        )

        img_compressed.save(
            os.path.join(filedirb, '{}_{}.{}'.format(name, i, img_format))
        )

        n += 1
        local_local_j += 1

        if local_local_j >= max_n:
            break

    return n


def main(target_dataset_folder, dataset_path, crop_size, img_format, max_patches, max_n, resize):
    print('[ Creating Dataset ]')
    print('Crop Size : {}'.format(crop_size))
    print('Target       : {}'.format(target_dataset_folder))
    print('Dataset       : {}'.format(dataset_path))
    print('Format    : {}'.format(img_format))
    print('Max N    : {}'.format(max_n))
    print('Resize factor    : {}'.format(resize))

    src_path = dataset_path
    if not dir_exists(src_path):
        raise(RuntimeError('Source folder not found, please put your dataset there'))

    set_path = target_dataset_folder

    mkdir(set_path)

    img_files = os.listdir(src_path)

    max = len(img_files)
    bar = tqdm(img_files)
    i = 0
    j = 0
    for files in bar:
        k = generate_patches(src_path, files, set_path,
                             crop_size, img_format, max_patches, resize, local_j=j, max_n=max_n)

        bar.set_description(desc='itr: %d/%d' % (
            i, max
        ))

        j += k

        if j >= max_n:
            # Stop the process
            print('Dataset count has been fullfuled')
            break

        i += 1

    print('Dataset Created')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--target_dataset_folder', type=str,
                        help='target folder where image saved')
    parser.add_argument('--dataset_path', type=str,
                        help='target folder where image saved')
    parser.add_argument('--max_patches', type=int,
                        help='target folder where image saved')
    parser.add_argument('--max_n', type=int,
                        help='target folder where image saved', default=99999999)
    parser.add_argument('--crop_size', type=int,
                        help='crop size, -1 to save whole images')
    parser.add_argument('--img_format', type=str, help='image format e.g. png')
    parser.add_argument(
        '--resize', type=int, default=0, help='resize image to that factor, positive (+) for upsample, (-) for downsample')

    args = parser.parse_args()

    crop_size = [args.crop_size,
                 args.crop_size] if args.crop_size > 0 else None
    main(args.target_dataset_folder, args.dataset_path,
         crop_size, args.img_format, args.max_patches, args.max_n, args.resize)
