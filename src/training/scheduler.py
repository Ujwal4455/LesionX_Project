from torch.optim import lr_scheduler


def build_scheduler(optimizer, epochs):
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
