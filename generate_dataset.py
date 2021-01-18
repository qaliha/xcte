import argparse
import os
import numpy as np

from PIL import Image
from tqdm import tqdm

def mkdir(directory, mode=0o777):
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.chmod(directory, mode=mode)

def dir_exists(directory):
    return os.path.exists(directory)

def crop(img_arr, block_size):
    h_b, w_b = block_size
    v_splited = np.vsplit(img_arr, img_arr.shape[0]//h_b)
    h_splited = np.concatenate([np.hsplit(col, img_arr.shape[1]//w_b) for col in v_splited], 0)
    return h_splited

def generate_patches(src_path, files, set_path, crop_size, img_format, max_patches):
    img_path = os.path.join(src_path, files)
    img = Image.open(img_path).convert('RGB')

    name, _ = files.split('.')
    filedir = os.path.join(set_path, 'a')
    if not dir_exists(filedir):
        mkdir(filedir)

    img = np.array(img)
    h, w = img.shape[0], img.shape[1]

    if crop_size == None:
        img = np.copy(img)
        img_patches = np.expand_dims(img, 0)
    else:
        rem_h = (h % crop_size[0])
        rem_w = (w % crop_size[1])
        img = img[:h-rem_h, :w-rem_w]
        img_patches = crop(img, crop_size)
    
    # print('Cropped')

    for i in range(min(len(img_patches), max_patches)):
        img = Image.fromarray(img_patches[i])

        img.save(
            os.path.join(filedir, '{}_{}.{}'.format(name, i, img_format))
        )

def main(target_dataset_folder, dataset_path, crop_size, img_format, max_patches):
    print('[ Creating Dataset ]')
    print('Crop Size : {}'.format(crop_size))
    print('Target       : {}'.format(target_dataset_folder))
    print('Dataset       : {}'.format(dataset_path))
    print('Format    : {}'.format(img_format))

    src_path = dataset_path
    if not dir_exists(src_path):
        raise(RuntimeError('Source folder not found, please put your dataset there'))

    set_path = target_dataset_folder

    mkdir(set_path)

    img_files = os.listdir(src_path)

    max = len(img_files)
    bar = tqdm(img_files)
    i = 0
    for files in bar:
        generate_patches(src_path, files, set_path, crop_size, img_format, max_patches)

        bar.set_description(desc='itr: %d/%d' %(
            i, max
        ))
        i += 1

    print('Dataset Created')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--target_dataset_folder', type=str, help='target folder where image saved')
    parser.add_argument('--dataset_path', type=str, help='target folder where image saved')
    parser.add_argument('--max_patches', type=int, help='target folder where image saved')
    parser.add_argument('--crop_size', type=int, help='crop size, -1 to save whole images')
    parser.add_argument('--img_format', type=str, help='image format e.g. png')
    
    args = parser.parse_args()

    crop_size = [args.crop_size, args.crop_size] if args.crop_size > 0 else None 
    main(args.target_dataset_folder, args.dataset_path, crop_size, args.img_format, args.max_patches)