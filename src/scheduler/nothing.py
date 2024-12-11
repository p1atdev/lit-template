import torch.optim.lr_scheduler as lr_scheduler


class NothingSchduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(NothingSchduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        pass
