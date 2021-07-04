import sys
import os
import warnings
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import get_training_set, get_test_set, tensor2img, psnr
from networks import Model
from torchinfo import summary


warnings.filterwarnings("ignore")


def main(opt):
    root_path = "dataset/"

    train_set = get_training_set(root_path + opt.dataset)
    test_set = get_test_set(root_path + opt.dataset)

    training_data_loader = DataLoader(
        dataset=train_set, num_workers=4, batch_size=opt.batch, shuffle=True)
    testing_data_loader = DataLoader(
        dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    net = Model(device, opt.model, opt)

    summaryEncoder = summary(net, input_size=(opt.batch, 3, 256, 256))
    num_epoch = opt.nepoch + 1
    for epoch in range(1, num_epoch):
        data_len = len(training_data_loader)
        bar = tqdm(enumerate(training_data_loader, 1),
                   total=data_len, disable=opt.silent)

        t_loss = 0
        net.train()

        for iteration, batch in bar:
            clean = batch[0].to(device)
            compressed = batch[1].to(device)

            net.set_input(compressed, clean)
            losses = net.optimize()

            t_loss += losses
            if not opt.silent:
                bar.set_description(desc='itr: %d/%d [%3d/%3d] [Dec: %.8f] Training' % (
                    iteration, data_len, epoch, num_epoch - 1,
                    t_loss/max(1, iteration)
                ))

        data_len_test = len(testing_data_loader)
        bar_test = tqdm(enumerate(testing_data_loader, 1),
                        total=data_len_test, disable=opt.silent)

        net.eval()
        psnrs = list()
        for iteration, batch in bar_test:
            with torch.no_grad():
                clean = batch[0].to(device)
                compressed = batch[0].to(device)

                clean_reconstructed = net(compressed)

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

        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))

        model_out_path = "checkpoint/{}/net_{}.pth".format(
            opt.dataset, opt.name)

        torch.save(net, model_out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compressing')

    parser.add_argument('--model', required=True, help='model')

    parser.add_argument('--dataset', required=True, help='path')
    parser.add_argument('--name', required=True, help='name')
    parser.add_argument('--nepoch', type=int, default=50, help='#')
    parser.add_argument('--cuda', action='store_true', help='cuda')
    parser.add_argument('--silent', action='store_true', help='cuda')
    parser.add_argument('--lr', type=float, default=0.0002, help='lr')
    parser.add_argument('--criterion', default='criterion', help='loss')
    parser.add_argument('--batch', type=int, default=8, help='#')
    opt = parser.parse_args()

    main(opt)
