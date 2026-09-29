import torch.optim as optim


def build_optimizer(model, backbone_lr=1e-5, head_lr=1e-4, weight_decay=1e-4):
    params = [
        {"params": model.backbone.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": model.classifier.parameters(), "lr": head_lr, "weight_decay": weight_decay},
    ]
    return optim.AdamW(params, lr=head_lr, weight_decay=weight_decay)
