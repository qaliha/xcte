import argparse
import os
from src.utils.metric import psnr, ssim

from src.utils.utils import add_padding
from src.utils.tensor import save_img_version, tensor2img

import torch
from torchvision import transforms

from loader import is_image_file
from src.model import Model
from src.utils.image import load_img

parser = argparse.ArgumentParser(description='testing framework')
parser.add_argument('--checkpoint', required=True, help='checkpoint folder')
# parser.add_argument('--bit', type=int, required=True, help='bit len')
parser.add_argument('--name', required=True, help='model name')
parser.add_argument('--e', type=int, default=200, help='model epoch')
parser.add_argument('--a', type=float, default=.75,
                    help='initial alpha gate for encoder')
parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "net_{}_epoch_{}.pth".format(opt.name, opt.e)
image_dir = "datasets_test/datasets/a/"

model = Model(0.5, opt).to(device)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_dict'])

model.bit_size = checkpoint['bit']

print(checkpoint['logs'])
print(model.Encoder.connection_weights)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor()]

transform = transforms.Compose(transform_list)

model.Encoder.eval()
model.Generator.eval()

psnr_sum = 0
ssim_sum = 0

for image_name in image_filenames:
    with torch.no_grad():
        # get input image
        input = load_img(image_dir + image_name, resize=False)

        # transforms and other operation
        input = transform(input)
        input = input.unsqueeze(0).to(device)

        # input_padded, h, w = add_padding(input, 128)

        encoder_output = model.Encoder(input)

        compressed_image = model.compress(encoder_output.detach())

        expanded_image = model.Generator(compressed_image)

        # expanded_image_dec = expanded_image[:, :, :h, :w]
        # compressed_image_dec = compressed_image[:, :, :h, :w]
        expanded_image_dec = expanded_image
        compressed_image_dec = compressed_image

        input_img = tensor2img(input)
        expanded_img = tensor2img(expanded_image_dec)
        compressed_img = tensor2img(compressed_image_dec)

        _tmp_psnr_compressed = psnr(input_img, compressed_img)
        _tmp_ssim_compressed = ssim(compressed_img, input_img)

        _tmp_psnr_expanded = psnr(input_img, expanded_img)
        _tmp_ssim_expanded = ssim(expanded_img, input_img)

        psnr_sum += _tmp_psnr_expanded
        ssim_sum += _tmp_ssim_expanded

        print(_tmp_psnr_compressed, _tmp_ssim_compressed,
              _tmp_psnr_expanded, _tmp_ssim_expanded)

        if not os.path.exists("results"):
            os.makedirs("results")

        save_img_version(compressed_image_dec.detach().squeeze(0).cpu(
        ), "results/{}_{}_compressed_{}".format(opt.name, opt.e, image_name))
        save_img_version(expanded_image_dec.detach().squeeze(0).cpu(
        ), "results/{}_{}_expanded_{}".format(opt.name, opt.e, image_name))

        torch.save(
            model, "model_{}_{}_expanded.pth".format(opt.name, opt.e))

print(psnr_sum/max(1, len(image_filenames)))
print(ssim_sum/max(1, len(image_filenames)))
