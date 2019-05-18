import torch.nn.functional as F


def RMSE(output, target):
    return (F.mse_loss(output, target) ** 0.5).item()


def EVA(output, target):
    diff = output - target
    return (1 - diff.var() / target.var()).item()
