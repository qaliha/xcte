from src.losses.mdf import MDFLoss
from src.utils.image import load_img
from torchvision import transforms
from loader import is_image_file
import os
import shutil
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
from src.utils.utils import add_padding, load_checkpoint, initialize_parameters_kaiming
from src.scheduler import get_scheduler, update_learning_rate
from src.model import Model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from generate_dataset import dir_exists, mkdir
from numpy.core.numeric import Infinity
from torchinfo import summary

from arguments import get_arguments

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # Training settings
    opt = get_arguments()

    writer = SummaryWriter(log_dir='runs/'+opt.name)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    print(opt)

    print('===> Loading datasets')
    root_path = "dataset/"
    train_set = get_training_set(
        root_path + opt.dataset)
    test_set = get_test_set(root_path + opt.dataset)

    training_data_loader = DataLoader(
        dataset=train_set, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    validating_data_loader = DataLoader(
        dataset=test_set, num_workers=4, batch_size=opt.test_batch_size, shuffle=False)

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    tb_process = None
    ngrok_process = None
    if opt.tensorboard:
        print('===> Running tensorboard')
        tb_process, ngrok_process = tf.launch_tensorboard(
            opt.hookbin, auth_token=opt.auth_token)

    print('===> Building models')

    net = Model(opt=opt).to(device)
    # net.apply(initialize_parameters_kaiming)
    model_summary = summary(net, input_size=(opt.batch_size, 3, 128, 128))

    criterion = MDFLoss('./weights/mdf.pth', opt.cuda)

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    sch_encoder = get_scheduler(optimizer, opt, option='step', step_length=10)

    start_epoch = opt.epoch_count

    num_epoch = opt.nepoch + 1
    tmp_epoch = 0
    for epoch in range(start_epoch, num_epoch):
        if opt.epoch_limit > 0 and epoch > opt.epoch_limit:
            print('Process stopped because has react limit')
            break

        tmp_epoch = epoch
        local_train_logs_holder = list()

        # start training
        net.train()
        data_len = len(training_data_loader)
        bar_train = tqdm(enumerate(training_data_loader, 1),
                         total=data_len, disable=opt.silent)
        t_loss = 0
        for iteration, batch in bar_train:
            # Train with random cropped image
            ground = batch[0].to(device)
            input = batch[1].to(device)

            optimizer.zero_grad()  # make gradient zero

            # calculate gradients
            output = net(input)
            loss = criterion(ground, output)
            loss.backward()

            # update weights
            optimizer.step()

            t_loss += loss.item()

            if not opt.silent:
                bar_train.set_description(desc='itr: %d/%d [%3d/%3d] [L: %.8f] Training Network' % (
                    iteration, data_len, epoch, num_epoch -
                    1, t_loss/max(1, iteration)
                ))

            if iteration == data_len:
                local_train_logs_holder.append(t_loss/max(1, iteration))

            if opt.tensorboard and iteration % 100 == 0:
                # Writing live loss
                num = ((epoch - 1) * data_len) + iteration
                writer.add_scalar('Live/Loss', t_loss/max(1, iteration), num)

        update_learning_rate(sch_encoder, optimizer, show=True)

        # Testing
        psnrs = list()
        ssims = list()

        net.eval()

        data_len_test = len(validating_data_loader)
        bar_test = tqdm(enumerate(validating_data_loader, 1),
                        total=data_len_test, disable=opt.silent)
        r_intermedient = torch.randint(0, data_len_test, (1,)).numpy()[0]
        for iteration, batch in bar_test:
            with torch.no_grad():
                ground = batch[0].to(device)
                input = batch[1].to(device)

                output = net(input)

                if r_intermedient == (iteration-1):
                    if not os.path.exists("interm"):
                        os.mkdir("interm")

                    output_clamped = torch.clamp(output, 0, 1)
                    image_tensor = torchvision.utils.make_grid(
                        [input.detach().squeeze(0), output_clamped.detach().squeeze(0)])

                    if opt.tensorboard:
                        writer.add_image(
                            'testing_image_sample', image_tensor, epoch)

                    save_img_version(image_tensor.cpu(),
                                     'interm/{}.png'.format(epoch))

                input_img = tensor2img(input)
                ground_img = tensor2img(ground)

                _tmp_psnr = psnr(ground_img, input_img)
                _tmp_ssim = ssim(input_img, ground_img)

                psnrs.append(_tmp_psnr)
                ssims.append(_tmp_ssim)

                if not opt.silent:
                    bar_test.set_description(desc='itr: %d/%d [%3d/%3d] P: %.4fdb S: %.4f Testing Image' % (
                        iteration, data_len_test, epoch, num_epoch - 1,
                        _tmp_psnr, _tmp_ssim
                    ))

        mean_psnrs = np.ma.masked_invalid(psnrs).mean()
        mean_ssims = np.ma.masked_invalid(ssims).mean()

        if opt.tensorboard:
            # Writing loss per epoch
            writer.add_scalar('Epoch/Loss', local_train_logs_holder[0], epoch)

            writer.add_scalar('Epoch/PSNR',
                              mean_psnrs, epoch)
            writer.add_scalar('Epoch/SSIM',
                              mean_ssims, epoch)

        print('[%3d/%3d] P: %.4fdb S: %.4f <-- Average' % (
            epoch, num_epoch - 1,
            mean_psnrs, mean_ssims
        ))

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
                'model_dict': net.state_dict(),
            }

            torch.save(state, model_out_path)
            print("Checkpoint saved to {} as {}".format(
                "checkpoint" + opt.dataset, model_out_path))

    # calculate the results
    # image_dir = "datasets_test/datasets/a/"
    # image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

    # transform_list = [transforms.ToTensor()]

    # transform = transforms.Compose(transform_list)

    # net.Encoder.eval()
    # net.Generator.eval()

    # psnr_sum = 0
    # ssim_sum = 0

    # for image_name in image_filenames:
    #     with torch.no_grad():
    #         # get input image
    #         input = load_img(image_dir + image_name, resize=False)

    #         # transforms and other operation
    #         input = transform(input)
    #         input = input.unsqueeze(0).to(device)

    #         input_padded, h, w = add_padding(input, 128)

    #         encoder_output = net.Encoder(input_padded)

    #         compressed_image = net.compress(encoder_output.detach())

    #         expanded_image = net.Generator(compressed_image)

    #         expanded_image_dec = expanded_image[:, :, :h, :w]
    #         compressed_image_dec = compressed_image[:, :, :h, :w]

    #         input_img = tensor2img(input)
    #         expanded_img = tensor2img(expanded_image_dec)
    #         compressed_img = tensor2img(compressed_image_dec)

    #         _tmp_psnr = psnr(input_img, compressed_img)
    #         _tmp_ssim = ssim(compressed_img, input_img)

    #         _tmp_psnr_expanded = psnr(input_img, expanded_img)
    #         _tmp_ssim_expanded = ssim(expanded_img, input_img)

    #         psnr_sum += _tmp_psnr_expanded
    #         ssim_sum += _tmp_ssim_expanded

    #         print(_tmp_psnr, _tmp_ssim,
    #               _tmp_psnr_expanded, _tmp_ssim_expanded)

    #         if not os.path.exists("results"):
    #             os.makedirs("results")

    #         # save_img_version(compressed_image_dec.detach().squeeze(0).cpu(), "results/{}_{}_compressed_{}".format(opt.name, tmp_epoch, image_name))
    #         # save_img_version(expanded_image_dec.detach().squeeze(0).cpu(), "results/{}_{}_expanded_{}".format(opt.name, tmp_epoch, image_name))

    # validation_psnr_accuracy = psnr_sum/max(1, len(image_filenames))
    # validation_ssim_accuracy = ssim_sum/max(1, len(image_filenames))

    # # logged hyperparameters
    # # learning rate
    # # bit size
    # writer.add_hparams({
    #     'seed': opt.seed,
    #     'lr': opt.lr,
    #     'bits': opt.bit,
    #     'gpu': torch.cuda.get_device_name(0),
    #     # todo
    #     'n_blocks': opt.n_blocks,
    #     'n_feature': opt.n_feature,
    #     'padding': opt.padding,
    #     'normalization': opt.normalization,
    #     'activation': opt.activation,
    #     # end todo
    #     'batch_size': opt.batch_size,
    #     'initial_gate_w': opt.a,
    #     't_params_enc': summaryEncoder.total_params,
    #     't_params_dec': summaryGenerator.total_params,
    # }, {
    #     'hparam/p_accuracy': validation_psnr_accuracy,
    #     'hparam/s_accuracy': validation_ssim_accuracy,
    # }, {
    #     'seed': [2047],
    #     'lr': [0.0001],
    #     'bits': [2, 3, 4, 5],
    #     'n_blocks': [4, 5, 6],
    #     'n_feature': [16, 32, 48, 64],
    #     'padding': ['default'],
    #     'normalization': ['channel', 'group', 'batch', 'instance'],
    #     'activation': ['prelu', 'relu', 'leaky', 'relu6', 'rrelu'],
    #     'batch_size': [8],
    # })
