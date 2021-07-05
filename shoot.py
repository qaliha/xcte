import argparse
import os
from src.utils.metric import psnr, ssim
from src.utils.tensor import save_img_version, tensor2img
from src.utils.utils import add_padding

from torchvision import transforms
from src.model import Model
from src.utils.image import load_img
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='testing framework')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument('--input', required=True, help='target input file')
    parser.add_argument('--output', required=True, help='target output file')
    parser.add_argument('--cuda', action='store_true', help='use cuda')

    parser.add_argument('--n_blocks', default=6, type=int, help='')
    parser.add_argument('--n_feature', default=64, type=int, help='')
    parser.add_argument('--padding', default='default', type=str, help='')
    parser.add_argument('--normalization',
                        default='channel', type=str, help='')
    parser.add_argument('--activation', default='relu', type=str, help='')
    parser.add_argument('--a', type=float, default=.75, help='initial gate')
    opt = parser.parse_args()

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    model = Model(3, opt).to(device)

    checkpoint = torch.load(opt.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_dict'])

    model.bit_size = checkpoint['bit']

    transform_list = [transforms.ToTensor()]

    transform = transforms.Compose(transform_list)

    model.Encoder.eval()
    model.Generator.eval()

    with torch.no_grad():
        input = load_img(opt.input, resize=False)

        # transforms and other operation
        input = transform(input)
        input = input.unsqueeze(0).to(device)

        input_padded, h, w = add_padding(input, 128)

        encoder_output = model.Encoder(input_padded)

        compressed_image = model.compress(encoder_output.detach())

        expanded_image = model.Generator(compressed_image)

        expanded_image_dec = expanded_image[:, :, :h, :w]
        compressed_image_dec = compressed_image[:, :, :h, :w]

        input_img = tensor2img(input)
        expanded_img = tensor2img(expanded_image_dec)
        compressed_img = tensor2img(compressed_image_dec)

        _tmp_psnr_expanded = psnr(input_img, expanded_img)

        print(_tmp_psnr_expanded)

        if not os.path.exists("out"):
            os.makedirs("out")

        save_img_version(compressed_image_dec.detach().squeeze(0).cpu(
        ), "out/{}".format(opt.output))
        save_img_version(expanded_image_dec.detach().squeeze(0).cpu(
        ), "out/{}".format(opt.output))
