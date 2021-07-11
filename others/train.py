import sys
import os
import warnings
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import get_training_set, get_test_set, tensor2img, psnr, add_padding, is_image_file
from generate_dataset import _compress
from networks import Model
from torchinfo import summary
from PIL import Image

warnings.filterwarnings("ignore")


def load_img(filepath, resize=True):
    img = Image.open(filepath).convert('RGB')
    if resize:
        img = img.resize((256, 256), Image.BICUBIC)
    return img


def main(opt):
    root_path = "dataset/"

    # print seed
    print(torch.seed)
    if opt.cuda:
        print(torch.cuda.seed)

    train_set = get_training_set(root_path + opt.dataset)
    test_set = get_test_set(root_path + opt.dataset)

    training_data_loader = DataLoader(
        dataset=train_set, num_workers=4, batch_size=opt.batch, shuffle=True)
    testing_data_loader = DataLoader(
        dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    net = Model(device, opt.model, opt)

    summaryNet = summary(net, input_size=(opt.batch, 3, 128, 128))
    num_epoch = opt.nepoch + 1
    tmp_epoch = 0
    for epoch in range(1, num_epoch):
        tmp_epoch = epoch

        data_len = len(training_data_loader)
        bar = tqdm(enumerate(training_data_loader, 1),
                   total=data_len, disable=opt.silent)

        t_loss = 0
        net.train()

        for iteration, batch in bar:
            clean = batch[0].to(device)
            halftoned = batch[1].to(device)

            net.set_input(halftoned, clean)
            losses = net.optimize()

            t_loss += losses
            if not opt.silent:
                bar.set_description(desc='itr: %d/%d [%3d/%3d] [Dec: %.8f] Training' % (
                    iteration, data_len, epoch, num_epoch - 1,
                    t_loss/max(1, iteration)
                ))

        # print loss
        print(t_loss/max(1, iteration))

        data_len_test = len(testing_data_loader)
        bar_test = tqdm(enumerate(testing_data_loader, 1),
                        total=data_len_test, disable=opt.silent)

        net.eval()
        psnrs = list()
        for iteration, batch in bar_test:
            with torch.no_grad():
                clean = batch[0].to(device)
                halftoned = batch[0].to(device)

                clean_reconstructed = net(halftoned)

                clean_img = tensor2img(clean)
                clean_reconstructed_img = tensor2img(clean_reconstructed)

                tmp_psnr = psnr(clean_img, clean_reconstructed_img)

                psnrs.append(tmp_psnr)

                if not opt.silent:
                    bar_test.set_description(desc='itr: %d/%d [%3d/%3d] PSNR: %.4fdb Testing Image' % (
                        iteration, data_len_test, epoch, num_epoch - 1,
                        tmp_psnr
                    ))

        mean_compressiong_psnr = np.ma.masked_invalid(psnrs).mean()
        print('[%3d/%3d] PSNR: %.4fdb <-- Average' % (
            epoch, num_epoch - 1,
            mean_compressiong_psnr
        ))

        # update learning rate
        net.step_scheduler()

        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))

        model_out_path = "checkpoint/{}/net_{}.pth".format(
            opt.dataset, opt.name)

        torch.save(net, model_out_path)

    if opt.model == 'mod_resblocks':
        # train pixCNN after training denoising model
        pixcnn_max_epochs = 10
        pix = Model(device, 'pixcnn', opt)

        summaryPix = summary(pix, input_size=(opt.batch, 3, 128, 128))

        num_epoch = pixcnn_max_epochs + 1
        tmp_epoch = 0
        for epoch in range(1, num_epoch):
            tmp_epoch = epoch

            data_len = len(training_data_loader)
            bar = tqdm(enumerate(training_data_loader, 1),
                       total=data_len, disable=opt.silent)

            t_loss = 0
            pix.train()

            # set requires grad False for net
            net.set_requires_grad_cs(False)
            net.eval()
            for iteration, batch in bar:
                clean = batch[0].to(device)
                halftoned = batch[1].to(device)

                reconstructed = net(halftoned)
                pix.set_input(reconstructed, clean)
                losses = pix.optimize()

                t_loss += losses
                if not opt.silent:
                    bar.set_description(desc='itr: %d/%d [%3d/%3d] [CorrectedLoss: %.8f] Training Pix' % (
                        iteration, data_len, epoch, num_epoch - 1,
                        t_loss/max(1, iteration)
                    ))

            # print loss
            print(t_loss/max(1, iteration))

            data_len_test = len(testing_data_loader)
            bar_test = tqdm(enumerate(testing_data_loader, 1),
                            total=data_len_test, disable=opt.silent)

            pix.eval()
            psnrs = list()
            for iteration, batch in bar_test:
                with torch.no_grad():
                    clean = batch[0].to(device)
                    halftoned = batch[0].to(device)

                    clean_reconstructed = net(halftoned)
                    clean_reconstructed_corrected = pix(clean_reconstructed)

                    clean_img = tensor2img(clean)
                    clean_reconstructed_img = tensor2img(
                        clean_reconstructed_corrected)

                    tmp_psnr = psnr(clean_img, clean_reconstructed_img)

                    psnrs.append(tmp_psnr)

                    if not opt.silent:
                        bar_test.set_description(desc='itr: %d/%d [%3d/%3d] PSNR: %.4fdb Testing Image' % (
                            iteration, data_len_test, epoch, num_epoch - 1,
                            tmp_psnr
                        ))

            mean_compressiong_psnr = np.ma.masked_invalid(psnrs).mean()
            print('[%3d/%3d] Corrected PSNR: %.4fdb <-- Average' % (
                epoch, num_epoch - 1,
                mean_compressiong_psnr
            ))

            # update learning rate
            pix.step_scheduler()

            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
                os.mkdir(os.path.join("checkpoint", opt.dataset))

            model_out_path = "checkpoint/{}/net_{}_pix.pth".format(
                opt.dataset, opt.name)

            torch.save(net, model_out_path)

    if opt.model == 'unet':
        image_dir = "datasets_test/datasets/a/"
        image_filenames = [x for x in os.listdir(
            image_dir) if is_image_file(x)]

        transform_list = [transforms.ToTensor()]

        transform = transforms.Compose(transform_list)

        psnr_sum = 0
        for image_name in image_filenames:
            with torch.no_grad():
                # get input image
                input = load_img(image_dir + image_name, resize=False)

                # transforms and other operation
                input = transform(input)
                input = input.unsqueeze(0).to(device)

                input_padded, h, w = add_padding(input, 128)
                compressed_image = _compress(input_padded, opt.bit)

                expanded_image = net(compressed_image)

                expanded_image_dec = expanded_image[:, :, :h, :w]

                input_img = tensor2img(input)
                compressed_img = tensor2img(compressed_image)
                expanded_img = tensor2img(expanded_image_dec)

                _tmp_psnr_expanded = psnr(input_img, expanded_img)

                psnr_sum += _tmp_psnr_expanded

                if not os.path.exists("results"):
                    os.makedirs("results")

                compressed_img.save(
                    "results/{}_{}_compressed_{}".format(opt.name, tmp_epoch, image_name))
                expanded_img.save(
                    "results/{}_{}_expanded_{}".format(opt.name, tmp_epoch, image_name))

        validation_psnr_accuracy = psnr_sum/max(1, len(image_filenames))
        print(validation_psnr_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compressing')

    parser.add_argument('--model', required=True, help='model')

    parser.add_argument('--dataset', required=True, help='path')
    parser.add_argument('--bit', required=True, type=int, help='bit')
    parser.add_argument('--name', required=True, help='name')
    parser.add_argument('--nepoch', type=int, default=50, help='#')
    parser.add_argument('--cuda', action='store_true', help='cuda')
    parser.add_argument('--silent', action='store_true', help='cuda')
    parser.add_argument('--lr', type=float, default=0.0002, help='lr')
    parser.add_argument('--criterion', default='criterion', help='loss')
    parser.add_argument('--batch', type=int, default=8, help='#')
    opt = parser.parse_args()

    main(opt)
