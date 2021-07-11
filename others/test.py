import argparse
import torch
from torchvision import transforms
from networks import Model
from utils import tensor2img, psnr
from generate_dataset import sp_halftone
from PIL import Image


def load_img(filepath, resize=True):
    img = Image.open(filepath).convert('RGB')
    if resize:
        img = img.resize((256, 256), Image.BICUBIC)
    return img


def main(opt):
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    model = torch.load(opt.model, map_location=device)

    model.eval()

    corrector = None
    if opt.pix:
        print('using pix')
        corrector = torch.load(opt.pix, map_location=device)
        corrector.eval()

    transform_list = [transforms.ToTensor()]

    transform = transforms.Compose(transform_list)

    with torch.no_grad():
        input = load_img(opt.ins, resize=False)
        halftoned = load_img(opt.ins, resize=False)
        halftoned_doed = sp_halftone(halftoned)

        halftoned_doed = transform(halftoned_doed)
        halftoned_doed = halftoned_doed.unsqueeze(0).to(device)

        reconstructed = model(halftoned_doed)

        if corrector is not None:
            reconstructed = corrector(reconstructed)

        halftoned_doed_img = tensor2img(halftoned_doed)
        reconstructed_img = tensor2img(reconstructed)

        psnr_result = psnr(input, reconstructed_img)

        print(psnr_result)

        halftoned_doed_img.save(opt.out + '_halftoned.png')
        reconstructed_img.save(opt.out + '_reconstructed.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compressing')

    parser.add_argument('--model', required=True, help='model')
    parser.add_argument('--pix', help='pix model')
    parser.add_argument('--ins', required=True, help='path')
    parser.add_argument('--out', required=True, help='path')
    parser.add_argument('--cuda', action='store_true', help='cuda')

    opt = parser.parse_args()

    main(opt)
