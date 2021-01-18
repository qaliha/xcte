import argparse

import torch
import torch.optim as optim

from src.model import Model

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Compressing')
    parser.add_argument('--dataset', required=True, help='dataset path')
    parser.add_argument('--name', required=True, help='training name')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--nepoch', type=int, default=50, help='# of epoch')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--epochsave', type=int, default=50, help='test')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')

    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use')

    opt = parser.parse_args()

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    
    print(opt)

# model = Model(bit=3)

# opt_encoder = optim.Adam(model.Encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
# opt_generator = optim.Adam(model.Generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# opt_discriminator = optim.Adam(model.Discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# input_image = torch.randn((1, 3, 256, 256))

# discriminator_loss, generator_losses = model.compression_forward(input_image)

# opt_generator.zero_grad()
# generator_losses.backward()
# opt_generator.step()

# opt_discriminator.zero_grad()
# discriminator_loss.backward()
# opt_discriminator.step()

# assert(list(model.Encoder.parameters())[0].grad is None)
# assert(list(model.Generator.parameters())[0].grad is not None)
# assert(list(model.Discriminator.parameters())[0].grad is not None)

# print(list(model.Encoder.parameters())[0].clone()[0][0])

# compression_losses = model.decompression_forward(input_image)

# opt_encoder.zero_grad()
# compression_losses.backward()
# opt_encoder.step()

# print(list(model.Encoder.parameters())[0].clone()[0][0])

# assert(list(model.Encoder.parameters())[0].grad is not None)
# assert(list(model.Generator.parameters())[0].grad is not None)
# assert(list(model.Discriminator.parameters())[0].grad is not None)

# print('Train?')