from torch.optim import lr_scheduler


def get_scheduler(optimizer, opt):
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + opt.epoch_count - 3) / \
            float(opt.nepoch - 3 + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    return scheduler

# update learning rate (called once every epoch)


def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('LR = %.7f' % lr)
