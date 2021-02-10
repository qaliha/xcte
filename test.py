import argparse
import os
from src.utils.tensor import prepare_for_compression_from_normalized_input, save_img_version, tensor2img

import torch
from torchvision import transforms

from loader import is_image_file
from src.model import Model
from src.utils.image import load_img

parser = argparse.ArgumentParser(description='testing framework')
parser.add_argument('--checkpoint', required=True, help='checkpoint folder')
parser.add_argument('--bit', type=int, required=True, help='bit len')
parser.add_argument('--name', required=True, help='model name')
parser.add_argument('--e', type=int, default=200, help='model epoch')
parser.add_argument('--a', type=float, default=.1,
                    help='initial alpha gate for encoder')
parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "checkpoints/{}/net_{}_epoch_{}.pth".format(
    opt.checkpoint, opt.name, opt.e)
image_dir = "checkpoints/{}/datasets/a/".format(opt.checkpoint)

model = Model(opt.bit, opt)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_dict'])

print(checkpoint['logs'])
print(model.Encoder.connection_weights)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)
for image_name in image_filenames:
    # get input image
    input = load_img(image_dir + image_name, resize=False)

    # transforms and other operation
    input = transform(input)
    input = input.unsqueeze(0).to(device)

    encoder_output = model.Encoder(input)

    compressed_image = model.compress(prepare_for_compression_from_normalized_input(
        encoder_output.detach().squeeze(0).cpu()))
    compressed_image = tensor2img(compressed_image)
    compressed_image = transforms.ToTensor()(compressed_image)

    # then normalize the image
    compressed_image_normalized = transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(compressed_image)
    # Add batch size and attach to device
    compressed_image_normalized = compressed_image_normalized.unsqueeze(
        0).to(device)

    # compressed_image_normalized = normalize(compressed_image)
    expanded_image = model.Generator(compressed_image_normalized)

    if not os.path.exists("checkpoints/{}/results".format(opt.checkpoint)):
        os.makedirs("checkpoints/{}/results".format(opt.checkpoint))

    save_img_version(compressed_image_normalized.detach().squeeze(0).cpu(
    ), "checkpoints/{}/results/{}_{}_compressed_{}".format(opt.checkpoint, opt.name, opt.e, image_name))
    save_img_version(expanded_image.detach().squeeze(0).cpu(
    ), "checkpoints/{}/results/{}_{}_expanded_{}".format(opt.checkpoint, opt.name, opt.e, image_name))
