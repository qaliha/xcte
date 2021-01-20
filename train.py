import os
import argparse
import random

from torchsummary import summary
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np

from src.model import Model
from src.scheduler import get_scheduler, update_learning_rate
from src.utils.tensor import normalize_input_from_normalied, prepare_for_compression_from_normalized_input, save_img, save_img_version, tensor2img
from src.utils.metric import psnr, ssim

# from loader import normalize
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
    parser.add_argument('--debug', action='store_true', help='use debug mode?')
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
    
    compression_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, shuffle=True)
    training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=opt.test_batch_size, shuffle=False)

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    print('===> Building models')

    model = Model(bit=opt.bit, opt=opt).to(device)

    summary(model.Encoder, (3, 256, 256), opt.batch_size)
    summary(model.Generator, (3, 256, 256), opt.batch_size)

    opt_encoder = optim.Adam(model.Encoder.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    opt_generator = optim.Adam(model.Generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    opt_discriminator = optim.Adam(model.Discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    sch_encoder = get_scheduler(opt_encoder, opt)
    sch_generator = get_scheduler(opt_generator, opt)
    sch_discriminator = get_scheduler(opt_discriminator, opt)

    start_epoch = opt.epoch_count
    if start_epoch > 1:
        pass
    
    parameters_encoder_bt = list(model.Encoder.parameters())[0].clone()
    parameters_generator_bt = list(model.Generator.parameters())[0].clone()
    parameters_discriminator_bt = list(model.Discriminator.parameters())[0].clone()

    num_epoch = opt.nepoch + 1
    for epoch in range(start_epoch, num_epoch):
        # starting the epoch

        data_len = len(compression_data_loader)
        bar = tqdm(enumerate(compression_data_loader, 1), total=data_len)

        # list of compressed image genrated by Encoder
        # compressed_images = list()
        # batched_images = list()

        model.Encoder.eval()
        for iteration, batch in bar:
            # compressing image
            image = batch[0].to(device)
            compressed_path = batch[2][0]

            # Convert to tensor first
            # compressed = _compress(image, 3)

            model.set_requires_grad(model.Encoder, False)

            # Encoder [-1, 1], Compressed: [0, 1]
            encoder_output = model.Encoder(image)
            compressed_image = model.compress(prepare_for_compression_from_normalized_input(encoder_output.detach().squeeze(0).cpu()))
            
            # save_img_version(image.detach().squeeze(0).cpu(), 'interm/encoder.png')

            # if iteration == 1:
            #     save_img(image[0].squeeze(0), 'in.png')
            #     save_img(compressed_image[0].squeeze(0), 'try.png')

            # if opt.debug:
            save_img(compressed_image, compressed_path)

            # compressed_images.append(compressed_image)
            # batched_images.append(image)

            # Save the compressed image to local disk
            # [-1., 1.] -> [0., 1.] -> *255

            # Make sure not requires gradient
            assert(compressed_image.requires_grad == False)

            bar.set_description(desc='itr: %d/%d [%3d/%3d] Compressing Image' %(
                iteration, data_len, epoch, num_epoch - 1
            ))


        data_len = len(training_data_loader)
        bar_ex = tqdm(enumerate(training_data_loader, 1), total=data_len)

        t_discriminator_loss = 0
        t_generator_losses = 0

        model.Generator.train()
        model.Discriminator.train()
        # Updating generator and discriminator parameters here
        for iteration, batch in bar_ex:
            # try to expanding the image
            image = batch[0].to(device)
            # assert(isinstance(batch[1], list) == False)

            compressed_image = batch[1].to(device)
            # compressed_image = compressed_images[iteration-1]

            # save_img_version(image.detach().squeeze(0).cpu(), 'interm/input.png')
            if opt.debug:
                save_img_version(compressed_image.detach().squeeze(0).cpu(), 'interm/encoder.png')

            # Normalize compressed image from [0, 1] to [-1, 0]
            # compressed_image = 2 * compressed_image  - 1

            assert(compressed_image is not None)
            assert(compressed_image.requires_grad == False)

            model.set_requires_grad(model.Generator, True)

            expanded = model.Generator(compressed_image)

            model.set_requires_grad(model.Discriminator, True)
            opt_discriminator.zero_grad()

            # Update discriminator
            fake_ab = model.Discriminator(torch.cat((compressed_image, expanded), 1).detach())
            loss_d_fake = model.gan_loss(fake_ab, False)

            real_ab = model.Discriminator(torch.cat((compressed_image, image), 1))
            loss_d_real = model.gan_loss(real_ab, True)

            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            discriminator_loss.backward()
            opt_discriminator.step()

            model.set_requires_grad(model.Discriminator, False)
            opt_generator.zero_grad()

            # Update generator
            fake_ab = model.Discriminator(torch.cat((compressed_image, expanded), 1))

            gan_losses = model.gan_loss(fake_ab, True)
            decoder_losses = model.restruction_loss(expanded, image)

            # perceptual_losses = model.perceptual_loss(expanded, image)

            # assert(expanded.requires_grad)
            # assert(image.requires_grad)

            if opt.debug:
                save_img_version(expanded.detach().squeeze(0).cpu(), 'interm/generated.png')
                save_img_version(image.detach().squeeze(0).cpu(), 'interm/inputed.png')
                save_img_version(compressed_image.detach().squeeze(0).cpu(), 'interm/compress.png')

            generator_losses = gan_losses + decoder_losses

            # discriminator_loss, generator_losses = model.gd_training(compressed_image, image)

            # Backward losses and step optimizer
            # Zero gradient
            
            generator_losses.backward()
            opt_generator.step()

            t_discriminator_loss += discriminator_loss.item()
            t_generator_losses += generator_losses.item()

            # assert(list(model.Encoder.parameters())[0].grad is None)
            # assert(list(model.Generator.parameters())[0].grad is not None)
            # assert(list(model.Discriminator.parameters())[0].grad is not None)

            bar_ex.set_description(desc='itr: %d/%d [%3d/%3d] [D_Loss: %.6f] [G_Loss: %.6f] Training Generator' %(
                iteration, data_len, epoch, num_epoch - 1,
                t_discriminator_loss/max(1, iteration),
                t_generator_losses/max(1, iteration)
            ))

        bar_enc = tqdm(enumerate(training_data_loader, 1), total=data_len)

        t_compression_losses = 0

        # Updating encoding parameters here
        model.Encoder.train()
        model.Generator.eval()
        for iteration, batch in bar_enc:
            # Original image
            image = batch[0].to(device)

            model.set_requires_grad(model.Generator, False)
            model.set_requires_grad(model.Encoder, True)

            opt_encoder.zero_grad()

            x = model.Encoder(image)
            # x = model.compress(x)
            
            if opt.debug:
                save_img_version(x.detach().squeeze(0).cpu(), 'interm/encoder.png')

            # Normalize the output first
            # x = normalize(x)
            x = model.Generator(x)

            compression_losses = model.compression_loss(x, image)

            if opt.debug:
                save_img_version(image.detach().squeeze(0).cpu(), 'interm/inputed.png')
                save_img_version(x.detach().squeeze(0).cpu(), 'interm/generated.png')

            # compression_losses = model.e_train(image)

            compression_losses.backward()
            opt_encoder.step()

            if opt.debug:
                print(model.Encoder.connection_weights)

            t_compression_losses += compression_losses.item()

            # assert(list(model.Encoder.parameters())[0].grad is not None)

            bar_enc.set_description(desc='itr: %d/%d [%3d/%3d] [E_Loss: %.6f] Training Encoder' %(
                iteration, data_len, epoch, num_epoch - 1,
                t_compression_losses/max(1, iteration)
            ))

        update_learning_rate(sch_encoder, opt_encoder)
        update_learning_rate(sch_generator, opt_generator)
        update_learning_rate(sch_discriminator, opt_discriminator)

        parameters_generator_af = list(model.Generator.parameters())[0].clone()
        parameters_discriminator_af = list(model.Discriminator.parameters())[0].clone()
        parameters_encoder_af = list(model.Encoder.parameters())[0].clone()

        assert(torch.all(torch.eq(parameters_generator_bt, parameters_generator_af)) == False)
        assert(torch.all(torch.eq(parameters_discriminator_bt, parameters_discriminator_af)) == False)
        assert(torch.all(torch.eq(parameters_encoder_bt, parameters_encoder_af)) == False)

        # Check custom parameter
        # print(model.Encoder.connection_weights)

        # Testing
        psnr_lists = list()
        ssim_lists = list()

        psnr_enc_lists = list()
        ssim_enc_lists = list()

        model.Encoder.eval()
        model.Generator.eval()

        model.set_requires_grad(model.Encoder, False)
        model.set_requires_grad(model.Generator, False)

        count_inf = 0

        data_len_test = len(testing_data_loader)
        bar_test = tqdm(enumerate(testing_data_loader, 1), total=data_len_test)
        r_intermedient = random.randint(0, data_len_test)
        for iteration, batch in bar_test:
            input = batch[0].to(device)

            encoder_output = model.Encoder(input)

            # compress the image from encoder
            compressed_image = model.compress(prepare_for_compression_from_normalized_input(encoder_output.detach().squeeze(0).cpu()))
            
            # Output from compress is [0, 1], before wi normalize, we must normalize this tensor to [-1, 0] first,
            # we can convert this to image and back to tensor
            compressed_image = tensor2img(compressed_image)
            compressed_image = transforms.ToTensor()(compressed_image)

            # then normalize the image
            compressed_image_normalized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(compressed_image)
            # Add batch size and attach to device
            compressed_image_normalized = compressed_image_normalized.unsqueeze(0).to(device)

            # compressed_image_normalized = normalize(compressed_image)
            expanded_image = model.Generator(compressed_image_normalized)
            
            if r_intermedient == (iteration-1):
                if not os.path.exists("interm"):
                    os.mkdir("interm")

                save_img_version(input.detach().squeeze(0).cpu(), 'interm/{}_input.png'.format(epoch))
                save_img_version(compressed_image_normalized.detach().squeeze(0).cpu(), 'interm/{}_compressed.png'.format(epoch))
                save_img_version(expanded_image.detach().squeeze(0).cpu(), 'interm/{}_expanded.png'.format(epoch))

            input_img = normalize_input_from_normalied(input.detach().squeeze(0).cpu())
            compressed_img = normalize_input_from_normalied(compressed_image_normalized.detach().squeeze(0).cpu())
            expanded_img = normalize_input_from_normalied(expanded_image.detach().squeeze(0).cpu())

            _tmp_psnr_compressed = psnr(input_img, compressed_img)
            _tmp_ssim_compressed = ssim(compressed_img, input_img)

            _tmp_psnr_expanded = psnr(input_img, expanded_img)
            _tmp_ssim_expanded = ssim(expanded_img, input_img)

            if _tmp_psnr_compressed:
                count_inf += 1

            if _tmp_psnr_expanded:
                count_inf += 1

            psnr_lists.append(_tmp_psnr_expanded)
            ssim_lists.append(_tmp_ssim_expanded)

            psnr_enc_lists.append(_tmp_psnr_compressed)
            ssim_enc_lists.append(_tmp_ssim_compressed)

            bar_test.set_description(desc='itr: %d/%d [%3d/%3d] C[P: %.4fdb S: %.4f] E[P: %.4fdb S: %.4f] Testing Image' %(
                iteration, data_len_test, epoch, num_epoch - 1,
                _tmp_psnr_compressed, _tmp_ssim_compressed,
                _tmp_psnr_expanded, _tmp_ssim_expanded
            ))

        print('[%3d/%3d] C[P: %.4fdb S: %.4f] E[P: %.4fdb S: %.4f] [Inf: %d] <-- Average' %(
            epoch, num_epoch - 1,
            np.ma.masked_invalid(psnr_enc_lists).mean(), np.ma.masked_invalid(ssim_enc_lists).mean(),
            np.ma.masked_invalid(psnr_lists).mean(), np.ma.masked_invalid(ssim_lists).mean(),
            count_inf
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