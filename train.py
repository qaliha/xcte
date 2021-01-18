import os
import argparse
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from src.model import Model
from src.scheduler import get_scheduler, update_learning_rate
from src.utils.tensor import save_img, tensor2img
from src.utils.metric import psnr, ssim

from loader_data import get_test_set, get_training_set
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Compressing')
    parser.add_argument('--dataset', required=True, help='dataset path')
    parser.add_argument('--bit', required=True, type=int, help='compression bit')
    parser.add_argument('--name', required=True, help='training name')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--nepoch', type=int, default=50, help='# of epoch')
    # parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    # parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--epochsave', type=int, default=50, help='test')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')

    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use')

    opt = parser.parse_args()

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    print(opt)

    print('===> Loading datasets')
    root_path = "dataset/"
    train_set = get_training_set(root_path + opt.dataset)
    test_set = get_test_set(root_path + opt.dataset)
    
    training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=opt.test_batch_size, shuffle=False)

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    print('===> Building models')

    model = Model(bit=opt.bit, opt=opt).to(device)

    opt_encoder = optim.Adam(model.Encoder.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    opt_generator = optim.Adam(model.Generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    opt_discriminator = optim.Adam(model.Discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    sch_encoder = get_scheduler(opt_encoder, opt)
    sch_generator = get_scheduler(opt_generator, opt)
    sch_discriminator = get_scheduler(opt_discriminator, opt)

    start_epoch = opt.epoch_count
    if start_epoch > 1:
        pass

    num_epoch = opt.nepoch + 1
    for epoch in range(start_epoch, num_epoch):
        # starting the epoch

        data_len = len(training_data_loader)
        bar = tqdm(enumerate(training_data_loader, 1), total=data_len)

        # list of compressed image genrated by Encoder
        compressed_images = list()

        model.Encoder.eval()
        for iteration, batch in bar:
            # compressing image
            image = batch.to(device)

            compressed_image = model.compression_forward_eval(image).detach()

            # if iteration == 1:
            #     save_img(image[0].squeeze(0), 'in.png')
            #     save_img(compressed_image[0].squeeze(0), 'try.png')

            compressed_images.append(compressed_image)
            # Make sure not requires gradient
            assert(compressed_image.requires_grad == False)

            bar.set_description(desc='itr: %d/%d [%3d/%3d] Compressing Image' %(
                iteration, data_len, epoch, num_epoch - 1
            ))

        bar_ex = tqdm(enumerate(training_data_loader, 1), total=data_len)

        model.Generator.train()
        model.Discriminator.train()
        # Updating generator and discriminator parameters here
        for iteration, batch in bar_ex:
            # try to expanding the image
            image = batch.to(device)
            compressed_image = compressed_images[iteration-1]

            # Zero gradient
            opt_generator.zero_grad()
            opt_discriminator.zero_grad()

            discriminator_loss, generator_losses = model.gd_training(compressed_image, image)

            # Backward losses and step optimizer
            generator_losses.backward()
            opt_generator.step()

            discriminator_loss.backward()
            opt_discriminator.step()

            bar_ex.set_description(desc='itr: %d/%d [%3d/%3d] [D_Loss: %.6f] [G_Loss: %.6f] Training Generator' %(
                iteration, data_len, epoch, num_epoch - 1,
                discriminator_loss.item(),
                generator_losses.item()
            ))

        bar_enc = tqdm(enumerate(training_data_loader, 1), total=data_len)

        # Updating encoding parameters here
        model.Encoder.train()
        model.Generator.eval()
        for iteration, batch in bar_enc:
            # Original image
            image = batch.to(device)

            opt_encoder.zero_grad()

            compression_losses = model.e_train(image)

            compression_losses.backward()
            opt_encoder.step()

            bar_enc.set_description(desc='itr: %d/%d [%3d/%3d] [E_Loss: %.6f] Training Encoder' %(
                iteration, data_len, epoch, num_epoch - 1,
                compression_losses.item()
            ))

        update_learning_rate(sch_encoder, opt_encoder)
        update_learning_rate(sch_generator, opt_generator)
        update_learning_rate(sch_discriminator, opt_discriminator)

        # Testing
        psnr_lists = list()
        ssim_lists = list()

        psnr_enc_lists = list()
        ssim_enc_lists = list()

        model.Encoder.eval()
        model.Generator.eval()

        data_len_test = len(testing_data_loader)
        bar_test = tqdm(enumerate(testing_data_loader, 1), total=data_len_test)
        r_intermedient = random.randint(0, data_len_test)
        for iteration, batch in bar_test:
            input = batch.to(device)

            with torch.no_grad():
                compressed_image = model.compression_forward_eval(input)
                expanded_image = model.Generator(compressed_image)
            
            if r_intermedient == (iteration-1):
                if not os.path.exists("intermediet"):
                    os.mkdir("intermediet")

                save_img(input.detach().squeeze(0).cpu(), '{}_input.png'.format(epoch))
                save_img(compressed_image.detach().squeeze(0).cpu(), '{}_compressed.png'.format(epoch))
                save_img(expanded_image.detach().squeeze(0).cpu(), '{}_expanded.png'.format(epoch))

            input_img = tensor2img(input)
            compressed_img = tensor2img(compressed_image)
            expanded_img = tensor2img(expanded_image)

            _tmp_psnr_compressed = psnr(input_img, compressed_img)
            _tmp_ssim_compressed = ssim(compressed_img, input_img)

            _tmp_psnr_expanded = psnr(input_img, expanded_img)
            _tmp_ssim_expanded = ssim(expanded_img, input_img)

            psnr_lists.append(_tmp_psnr_expanded)
            ssim_lists.append(_tmp_ssim_expanded)

            psnr_enc_lists.append(_tmp_psnr_compressed)
            ssim_enc_lists.append(_tmp_ssim_compressed)

            bar_test.set_description(desc='itr: %d/%d [%3d/%3d] C[P: %.4fdb S: %.4f] E[P: %.4fdb S: %.4f] Testing Image' %(
                iteration, data_len, epoch, num_epoch - 1,
                _tmp_psnr_compressed, _tmp_ssim_compressed,
                _tmp_psnr_expanded, _tmp_ssim_expanded
            ))

        print('[%3d/%3d] C[P: %.4fdb S: %.4f] E[P: %.4fdb S: %.4f] <-- Average' %(
            epoch, num_epoch - 1,
            np.mean(psnr_enc_lists), np.mean(ssim_enc_lists),
            np.mean(psnr_lists), np.mean(ssim_lists)
        ))

        # Generate checkpoint
        if epoch % opt.epochsave == 0:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
                os.mkdir(os.path.join("checkpoint", opt.dataset))

            model_out_path = "checkpoint/{}/net_{}_epoch_{}.pth".format(opt.dataset, opt.name, epoch)

            state = {
                'epoch': epoch + 1,
                'model_dict': model.state_dict()
            }

            torch.save(state, model_out_path)
            print("Checkpoint saved to {} as {}".format("checkpoint" + opt.dataset, model_out_path))