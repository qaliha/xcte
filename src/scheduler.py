from torch.optim import lr_scheduler


def get_scheduler(optimizer, opt, option='lambda', step_length=15):
    scheduler = None
    if option == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - 10) / float(opt.nepoch - 10)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif option == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=step_length, gamma=0.1)

    return scheduler

# update learning rate (called once every epoch)


def update_learning_rate(scheduler, optimizer, show=False):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    if show:
        print('LR Updated to = %.7f' % lr)
