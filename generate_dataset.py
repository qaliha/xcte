import os
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from sklearn.feature_extraction import image as GetPatches

from src.utils.utils import dir_exists, mkdir, sp_halftone


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--crop', type=int, default=0,
                        help='crop size, -1 to save whole images')
    parser.add_argument('--target', type=str, default='',
                        help='target folder where image saved')
    parser.add_argument('--path', type=str, default='',
                        help='source dataset folder')
    parser.add_argument('--max', type=int, default=1,
                        help='maximum patches per images')

    args = parser.parse_args()
    crop_size = [args.crop, args.crop] if args.crop > 0 else None

    print('Creating dataset...')

    src_path = args.path
    if not dir_exists(src_path):
        raise(RuntimeError('Source folder not found, please put your dataset there'))

    set_path = args.target
    mkdir(set_path)

    img_files = os.listdir(src_path)
    max_files = len(img_files)
    bar = tqdm(img_files)
    j = 0
    for files in bar:
        # Open image
        img_path = os.path.join(src_path, files)
        img = Image.open(img_path).convert('RGB')

        name, _ = files.split('.')
        filedir = os.path.join(set_path, 'a')
        if not dir_exists(filedir):
            mkdir(filedir)

        filedir_b = os.path.join(set_path, 'b')
        if not dir_exists(filedir_b):
            mkdir(filedir_b)

        img_array = np.array(img)
        if crop_size == None:
            img_array = np.copy(img_array)
            img_patches = np.expand_dims(img_array, 0)
        else:
            img_patches = GetPatches.extract_patches_2d(
                img_array, (crop_size[0], crop_size[1]), max_patches=args.max, random_state=0)

        for i in range(len(img_patches)):
            # Do halftoning
            img = Image.fromarray(img_patches[i])
            img_halftoned = sp_halftone(img.convert('RGB'))

            img.save(os.path.join(filedir, '{}_{}.{}'.format(name, i, 'png')))
            img_halftoned.save(os.path.join(
                filedir_b, '{}_{}.{}'.format(name, i, 'png')))

        bar.set_description(desc='itr: %d/%d' % (
            j, max_files
        ))

        j += 1

    print('Dataset created')
