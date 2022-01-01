from torch import nn
from torch.optim import Optimizer, Adam, AdamW, RMSprop
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR, StepLR, ReduceLROnPlateau


def choose_optimizer(name: str, model: nn.Module, lr: float) -> Optimizer:
    if name == 'SGD':
        return SGD(model.parameters(), lr=lr)

    elif name == 'Adam':
        return Adam(model.parameters(), lr=lr)

    elif name == 'AdamW':
        return AdamW(model.parameters(), lr=lr)

    elif name == 'RMSprop':
        return RMSprop(model.parameters(), lr=lr)

    else:
        raise NotImplemented(f"{name} optimizer not found!")


def choose_scheduler(name: str, optimizer: Optimizer, **kwargs):
    steps_per_epochs = kwargs['steps_per_epochs']
    epochs = kwargs['epochs']
    lr = kwargs['lr']

    if name == 'OneCycleLR':
        return OneCycleLR(optimizer,
                          epochs=epochs,
                          steps_per_epoch=steps_per_epochs,
                          max_lr=lr)

    elif name == 'StepLR':
        return StepLR(optimizer, step_size=epochs // 3, gamma=0.2)

    elif name == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.2)

    else:
        raise NotImplemented("{name} is not implemented!")
