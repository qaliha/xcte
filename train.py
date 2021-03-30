import os
import shutil
import random
import warnings
import numpy as np
import torch.optim as optim
import torch
import tf
import torchvision
from os.path import join
from tqdm import tqdm
from loader_data import get_test_set, get_training_set
from src.utils.metric import psnr, ssim
from src.utils.tensor import save_img, save_img_version, tensor2img
from src.utils.utils import initialize_parameters, load_checkpoint, normalize, inverse_normalize
from src.scheduler import get_scheduler, update_learning_rate
from src.model import Model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from generate_dataset import dir_exists, mkdir
from numpy.core.numeric import Infinity
from torchsummary import summary

from arguments import get_arguments

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    writer = SummaryWriter()

    # Training settings
    opt = get_arguments()

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    print(opt)

    print('===> Loading datasets')
    root_path = "dataset/"
    train_set = get_training_set(
        root_path + opt.dataset, scale_n_crop=opt.noscale == False)
    test_set = get_test_set(root_path + opt.dataset,
                            scale_n_crop=opt.noscale == False)

    training_data_loader = DataLoader(
        dataset=train_set, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    testing_data_loader = DataLoader(
        dataset=test_set, num_workers=4, batch_size=opt.test_batch_size, shuffle=False)

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    tb_process = None
    ngrok_process = None
    if opt.tensorboard:
        print('===> Running tensorboard')
        tb_process, ngrok_process = tf.launch_tensorboard(
            opt.hookbin, auth_token=opt.auth_token)

    print('===> Building models')

    model = Model(bit=opt.bit, opt=opt).to(device)

    model.apply(initialize_parameters)

    sample_image_dataset = training_data_loader.dataset
    n_samples = len(sample_image_dataset)

    # to get a random sample
    random_index = int(np.random.random()*n_samples)
    single_example = sample_image_dataset[random_index]
    single_example_size = list(single_example[0].size())
    single_example = (
        single_example_size[0], single_example_size[1], single_example_size[2])

    summary(model.Encoder, single_example, opt.batch_size)
    summary(model.Generator, single_example, opt.batch_size)

    opt_encoder = optim.Adam(model.Encoder.parameters(), lr=opt.lr)
    opt_generator = optim.Adam(model.Generator.parameters(), lr=opt.lr)
    opt_discriminator = optim.Adam(model.Discriminator.parameters(), lr=opt.lr)

    sch_encoder = get_scheduler(opt_encoder, opt, option='step')
    sch_generator = get_scheduler(opt_generator, opt, option='step')
    sch_discriminator = get_scheduler(opt_discriminator, opt, option='step')

    train_logs_holder = list()

    start_epoch = opt.epoch_count

    max_ssim = -Infinity
    max_ssim_epoch = 0

    tmp_bit_size = opt.bit

    if start_epoch > 1:
        start_epoch, model, opt_encoder, opt_generator, opt_discriminator, sch_encoder, sch_generator, sch_discriminator, train_logs_holder, max_ssim, max_ssim_epoch, tmp_bit_size = load_checkpoint(
            model, opt_encoder, opt_generator, opt_discriminator, sch_encoder, sch_generator, sch_discriminator, "checkpoint/{}/net_{}_epoch_{}.pth".format(opt.dataset, opt.name, start_epoch-1))

        print('Previous learning rate = {}'.format(
            opt_generator.param_groups[0]['lr']))

        model.bit_size = tmp_bit_size

        for state in opt_encoder.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        for state in opt_generator.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        for state in opt_discriminator.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        # We remove the b folders here
        train_dir_copy = join(root_path + opt.dataset, "train", "b")
        if dir_exists(train_dir_copy):
            shutil.rmtree(train_dir_copy)
            print('Training compressed image has been delted')

            mkdir(train_dir_copy)

    training_data_loader.dataset.set_load_compressed(False)  # for speedup

    # Calculating mean and standard deviation for dataset

    nimages = 0
    mean = 0.0
    var = 0.0
    for iteration, data in enumerate(training_data_loader, 1):
        batch = data[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    if opt.tensorboard:
        writer.add_text('logs', f"Found mean: {mean}, std: {std}", 0)
    else:
        print(f"Found mean: {mean}, std: {std}")

    training_mean = (*mean,)
    training_std = (*std,)

    num_epoch = opt.nepoch + 1
    for epoch in range(start_epoch, num_epoch):
        if opt.epoch_limit > 0 and epoch > opt.epoch_limit:
            print('Process stopped because has react limit')
            break

        # warming the parameters of encoder if --warm is provided
        if opt.warm and epoch == 1:
            t_warm_losses = 0

            # Temporary disable gradient for connection weights
            model.Encoder.connection_weights.requires_grad = False
            if opt.debug:
                print(model.Encoder.connection_weights)

            training_data_loader.dataset.set_load_compressed(
                False)  # for speedup

            data_len = len(training_data_loader)
            bar_enc = tqdm(enumerate(training_data_loader, 1),
                           total=data_len, disable=opt.silent)

            model.Encoder.train()
            for iteration, batch in bar_enc:
                # Train with random cropped image
                image = batch[0+3].to(device)

                opt_encoder.zero_grad()  # make gradient zero

                # calculate gradients
                encoded = model.Encoder(image)
                compression_losses = model.compression_loss(encoded, image)
                compression_losses.backward()

                # update weights
                opt_encoder.step()

                t_warm_losses += compression_losses.item()

                if opt.debug:
                    save_img_version(encoded.detach().squeeze(
                        0).cpu(), 'interm/warm.png')

                    print(model.Encoder.connection_weights)

                if not opt.silent:
                    bar_enc.set_description(desc='itr: %d/%d [%3d/%3d] [L: %.8f] Warming Encoder' % (
                        iteration, data_len, epoch, num_epoch -
                        1, t_warm_losses/max(1, iteration)
                    ))

            if opt.tensorboard:
                writer.add_text(
                    'logs', f'Warming loss: {t_warm_losses/max(1, data_len)}')
                writer.add_text(
                    'logs', f'Connection weights after training: {model.Encoder.connection_weights.item()}')

            # Re enable after the warming
            model.Encoder.connection_weights.requires_grad = True
            # if opt.debug:
            print(
                f'Connection weights after training: {model.Encoder.connection_weights.item()}')

        if opt.debug:
            print(model.Encoder.connection_weights)

        local_train_logs_holder = list()

        training_data_loader.dataset.set_load_compressed(
            False)  # don't load b for speedup
        data_len = len(training_data_loader)
        bar = tqdm(enumerate(training_data_loader, 1),
                   total=data_len, disable=opt.silent)

        if opt.tensorboard:
            writer.add_text(
                'logs', f'Epoch {epoch} - Compressing Image', epoch)

        model.Encoder.eval()

        for iteration, batch in bar:
            with torch.no_grad():
                # compress original image (not cropped to get full image)
                image = batch[0].to(device)
                compressed_path = batch[2]

                encoder_output = model.Encoder(image)

                compressed_image = model.compress(encoder_output.detach())

                for i in range(compressed_image.size(0)):
                    # Save the compressed image to local disk
                    # [-1., 1.] -> [0., 1.] -> *255
                    save_img(compressed_image[i, :, :, :], compressed_path[i])

                if not opt.silent:
                    bar.set_description(desc='itr: %d/%d [%3d/%3d] Compressing Image' % (
                        iteration, data_len, epoch, num_epoch - 1
                    ))

        training_data_loader.dataset.set_load_compressed(
            True)  # load compressed for training

        data_len = len(training_data_loader)
        bar_ex = tqdm(enumerate(training_data_loader, 1),
                      total=data_len, disable=opt.silent)

        t_discriminator_loss = 0
        t_generator_losses = 0
        t_dec_losses = 0

        if opt.tensorboard:
            writer.add_text(
                'logs', f'Epoch {epoch} - Training Generator', epoch)

        model.Generator.train()
        model.Discriminator.train()
        # Updating generator and discriminator parameters here
        for iteration, batch in bar_ex:
            # try to expanding the image
            image = batch[0+3].to(device)
            compressed_image = batch[1+3].to(device)

            # because gradients for generator disabled when training Encoder
            # here we make generator can calculate gradients again!
            model.set_requires_grad(model.Generator, True)

            # Train discriminator
            # enable backprop for discriminator
            model.set_requires_grad(model.Discriminator, True)
            opt_discriminator.zero_grad()  # make D's gradients zero

            # calculate gradients
            expanded = model.Generator(compressed_image)

            D_in = torch.cat((image, expanded), dim=1)
            D_out, D_out_logits = model.Discriminator(D_in.detach())
            D_out = torch.squeeze(D_out)
            D_out_logits = torch.squeeze(D_out_logits)

            D_real, D_gen = torch.chunk(D_out, 2, dim=0)
            D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)

            discriminator_loss = model.gan_loss_hf(
                D_real, D_gen, D_real_logits, D_gen_logits, 'discriminator_loss')

            discriminator_loss.backward()
            # update D's weights
            opt_discriminator.step()

            # D's required not gradients when optimizing G
            model.set_requires_grad(model.Discriminator, False)
            opt_generator.zero_grad()  # make G's gradients zero

            # calculating gradients
            D_out, D_out_logits = model.Discriminator(
                torch.cat((image, expanded), dim=1))
            D_out = torch.squeeze(D_out)
            D_out_logits = torch.squeeze(D_out_logits)

            D_real, D_gen = torch.chunk(D_out, 2, dim=0)
            D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)

            gan_losses = model.gan_loss_hf(
                D_real, D_gen, D_real_logits, D_gen_logits, 'generator_loss')

            decoder_losses = model.restruction_loss(expanded, image)
            generator_losses = gan_losses * 0.001 + decoder_losses

            generator_losses.backward()

            # update G's weights
            opt_generator.step()

            t_discriminator_loss += discriminator_loss.item()
            t_generator_losses += generator_losses.item()
            t_dec_losses += decoder_losses.item()

            if iteration == data_len:
                local_train_logs_holder.append(
                    t_discriminator_loss/max(1, iteration))
                local_train_logs_holder.append(
                    t_generator_losses/max(1, iteration))

            if opt.tensorboard and iteration % 100 == 0:
                # Writing live loss
                num = ((epoch - 1) * data_len) + iteration
                writer.add_scalar('LiveLoss/Discriminator', t_discriminator_loss /
                                  max(1, iteration), num)
                writer.add_scalar('LiveLoss/Generator', t_generator_losses /
                                  max(1, iteration), num)
                writer.add_scalar('LiveLoss/Decoder', t_dec_losses /
                                  max(1, iteration), num)

            if opt.debug:
                save_img_version(expanded.detach().squeeze(
                    0).cpu(), 'interm/generated.png')
                save_img_version(image.detach().squeeze(
                    0).cpu(), 'interm/inputed.png')
                save_img_version(compressed_image.detach().squeeze(
                    0).cpu(), 'interm/compress.png')

            if not opt.silent:
                bar_ex.set_description(desc='itr: %d/%d [%3d/%3d] [D: %.8f] [G: %.8f] [Dec: %.8f] Training Generator' % (
                    iteration, data_len, epoch, num_epoch - 1,
                    t_discriminator_loss/max(1, iteration),
                    t_generator_losses/max(1, iteration),
                    t_dec_losses/max(1, iteration)
                ))

        if opt.tensorboard:
            writer.add_text(
                'logs', f'Epoch {epoch} - Training Encoder', epoch)

        training_data_loader.dataset.set_load_compressed(True)  # for speedup
        bar_enc = tqdm(enumerate(training_data_loader, 1),
                       total=data_len, disable=opt.silent)

        t_compression_losses = 0

        # Updating encoding parameters here
        model.Encoder.train()
        model.Generator.train()
        for iteration, batch in bar_enc:
            # Get random cropped image
            image = batch[0+3].to(device)
            compressed_image = batch[1+3].to(device)

            # G requires no gradient when optimizing E
            model.set_requires_grad(model.Generator, False)

            opt_encoder.zero_grad()  # set E's gradients to zero

            # calculate gradient for E
            encoded = model.Encoder(image)

            # new method
            if opt.optimized_encoder:
                encoded = 0.5 * encoded + 0.5 * compressed_image

            # if opt.optimizer_encoder_noises:
            #     # possible image noise when compressing image
            #     noise = compressed_image - encoded.detach()
            #     encoded += noise

            generated = model.Generator(encoded)

            compression_losses = model.compression_loss(generated, image)
            compression_losses.backward()

            # update E's weights
            opt_encoder.step()

            t_compression_losses += compression_losses.item()

            if iteration == data_len:
                local_train_logs_holder.append(
                    t_compression_losses/max(1, iteration))

            if not opt.silent:
                bar_enc.set_description(desc='itr: %d/%d [%3d/%3d] [E: %.8f] Training Encoder' % (
                    iteration, data_len, epoch, num_epoch - 1,
                    t_compression_losses/max(1, iteration)
                ))

            if opt.tensorboard and iteration % 100 == 0:
                # Writing live loss
                num = ((epoch - 1) * data_len) + iteration
                writer.add_scalar('LiveLoss/Encoder', t_compression_losses /
                                  max(1, iteration), num)

            if opt.debug:
                save_img_version(encoded.detach().squeeze(
                    0).cpu(), 'interm/encoder.png')
                save_img_version(image.detach().squeeze(
                    0).cpu(), 'interm/inputed.png')
                save_img_version(generated.detach().squeeze(
                    0).cpu(), 'interm/generated.png')
                # save_img_version(encoded_masked.detach().squeeze(
                #     0).cpu(), 'interm/masked.png')

                print(model.Encoder.connection_weights)

        if opt.tensorboard:
            writer.add_text(
                'logs', f'Connection weights after training: {model.Encoder.connection_weights.item()}', epoch)

        print(
            f'Connection weights after training: {model.Encoder.connection_weights.item()}')

        local_train_logs_holder.append(model.Encoder.connection_weights.item())

        update_learning_rate(sch_encoder, opt_encoder, show=True)
        update_learning_rate(sch_generator, opt_generator)
        update_learning_rate(sch_discriminator, opt_discriminator)

        # Testing
        psnr_lists = list()
        ssim_lists = list()

        psnr_enc_lists = list()
        ssim_enc_lists = list()

        model.Encoder.eval()
        model.Generator.eval()

        count_inf = 0

        if opt.tensorboard:
            writer.add_text(
                'logs', f'Epoch {epoch} - Validation Model', epoch)

        testing_data_loader.dataset.set_load_compressed(False)  # for speed up

        data_len_test = len(testing_data_loader)
        bar_test = tqdm(enumerate(testing_data_loader, 1),
                        total=data_len_test, disable=opt.silent)
        r_intermedient = random.randint(0, data_len_test)
        for iteration, batch in bar_test:
            with torch.no_grad():
                input = batch[0+3].to(device)

                encoder_output = model.Encoder(input)

                # compress the image from encoder
                # compressed_image = model.compress(prepare_for_compression_from_normalized_input(
                # encoder_output.detach().squeeze(0).cpu()))

                compressed_image = model.compress(encoder_output.detach())

                expanded_image = model.Generator(compressed_image)

                if r_intermedient == (iteration-1):
                    if not os.path.exists("interm"):
                        os.mkdir("interm")

                    expanded_image_clamped = torch.clamp(expanded_image, 0, 1)
                    image_tensor = torchvision.utils.make_grid(
                        [input.detach().squeeze(0), compressed_image.detach().squeeze(0), expanded_image_clamped.detach().squeeze(0)])

                    if opt.tensorboard:
                        writer.add_image(
                            'testing_image_sample', image_tensor, epoch)

                    save_img_version(image_tensor.cpu(),
                                     'interm/{}.png'.format(epoch))

                input_img = tensor2img(input)
                compressed_img = tensor2img(compressed_image)
                expanded_img = tensor2img(expanded_image)

                _tmp_psnr_compressed = psnr(input_img, compressed_img)
                _tmp_ssim_compressed = ssim(compressed_img, input_img)

                _tmp_psnr_expanded = psnr(input_img, expanded_img)
                _tmp_ssim_expanded = ssim(expanded_img, input_img)

                if _tmp_psnr_compressed >= Infinity:
                    count_inf += 1

                if _tmp_psnr_expanded >= Infinity:
                    count_inf += 1

                psnr_lists.append(_tmp_psnr_expanded)
                ssim_lists.append(_tmp_ssim_expanded)

                psnr_enc_lists.append(_tmp_psnr_compressed)
                ssim_enc_lists.append(_tmp_ssim_compressed)

                if not opt.silent:
                    bar_test.set_description(desc='itr: %d/%d [%3d/%3d] C[P: %.4fdb S: %.4f] E[P: %.4fdb S: %.4f] Testing Image' % (
                        iteration, data_len_test, epoch, num_epoch - 1,
                        _tmp_psnr_compressed, _tmp_ssim_compressed,
                        _tmp_psnr_expanded, _tmp_ssim_expanded
                    ))

        mean_compressiong_psnr = np.ma.masked_invalid(psnr_enc_lists).mean()
        mean_compressing_ssim = np.ma.masked_invalid(ssim_enc_lists).mean()
        mean_expanding_psnr = np.ma.masked_invalid(psnr_lists).mean()
        mean_expanding_ssim = np.ma.masked_invalid(ssim_lists).mean()

        if opt.tensorboard:
            # Writing loss per epoch
            writer.add_scalar('Loss/Discriminator',
                              local_train_logs_holder[0], epoch)
            writer.add_scalar('Loss/Generator',
                              local_train_logs_holder[1], epoch)
            writer.add_scalar(
                'Loss/Encoder', local_train_logs_holder[2], epoch)
            writer.add_scalar('Parameters/Gate',
                              local_train_logs_holder[3], epoch)

            writer.add_scalar('Metrics/Compression/PSNR',
                              mean_compressiong_psnr, epoch)
            writer.add_scalar('Metrics/Compression/SSIM',
                              mean_compressing_ssim, epoch)
            writer.add_scalar('Metrics/Expansion/PSNR',
                              mean_expanding_psnr, epoch)
            writer.add_scalar('Metrics/Expansion/SSIM',
                              mean_expanding_ssim, epoch)

        print('[%3d/%3d] C[P: %.4fdb S: %.4f] E[P: %.4fdb S: %.4f] [Inf: %d] <-- Average' % (
            epoch, num_epoch - 1,
            mean_compressiong_psnr, mean_compressing_ssim,
            mean_expanding_psnr, mean_expanding_ssim,
            count_inf
        ))

        local_train_logs_holder.append(
            [mean_compressiong_psnr, mean_compressing_ssim, mean_expanding_psnr, mean_expanding_ssim])

        if opt.debug:
            print(local_train_logs_holder)

        train_logs_holder.append(local_train_logs_holder)

        # Generate checkpoint
        if epoch % opt.epochsave == 0:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
                os.mkdir(os.path.join("checkpoint", opt.dataset))

            model_out_old_path = "checkpoint/{}/net_{}_epoch_{}.pth".format(
                opt.dataset, opt.name, epoch - opt.epochsave)

            if opt.commit:
                if os.path.exists(model_out_old_path):
                    os.remove(model_out_old_path)

            model_out_path = "checkpoint/{}/net_{}_epoch_{}.pth".format(
                opt.dataset, opt.name, epoch)

            state = {
                'epoch': epoch + 1,
                # Model
                'model_dict': model.state_dict(),
                # Optimizer
                # 'optimizer_e': opt_encoder.state_dict() if opt.commit else None,
                # 'optimizer_g': opt_generator.state_dict() if opt.commit else None,
                # 'optimizer_d': opt_discriminator.state_dict() if opt.commit else None,
                # Scheduler
                # 'scheduler_e': sch_encoder.state_dict() if opt.commit else None,
                # 'scheduler_g': sch_generator.state_dict() if opt.commit else None,
                # 'scheduler_d': sch_discriminator.state_dict() if opt.commit else None,
                # Optimizer
                'optimizer_e': None,
                'optimizer_g': None,
                'optimizer_d': None,
                # Scheduler
                'scheduler_e': None,
                'scheduler_g': None,
                'scheduler_d': None,
                'logs': train_logs_holder,
                'max_ssim': max_ssim,
                'max_ssim_epoch': max_ssim_epoch,
                'bit': model.bit_size
            }

            if mean_expanding_ssim >= max_ssim:
                notice = f"Found new max SSIM on epoch {epoch}. {max_ssim} -> {mean_expanding_ssim}"
                writer.add_text('logs', notice, epoch)
                print(notice)

                max_ssim = mean_expanding_ssim
                max_ssim_epoch = epoch

                state['max_ssim'] = max_ssim
                state['max_ssim_epoch'] = max_ssim_epoch

                # are old file exist
                modelmax_old_path = "checkpoint/{}/net_max.pth".format(
                    opt.dataset)
                if os.path.exists(modelmax_old_path):
                    os.remove(modelmax_old_path)

                torch.save(state, modelmax_old_path)

            torch.save(state, model_out_path)
            print("Checkpoint saved to {} as {}".format(
                "checkpoint" + opt.dataset, model_out_path))

    # calculate the results
