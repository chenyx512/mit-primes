import torch.nn.functional as F
import torch


def RMSE(output, target):
    return (F.mse_loss(output, target) ** 0.5).item()


def EVA(output, target):
    """Note: this does not work with batch size 1"""
    diff = output - target
    return (1 - diff.var() / target.var()).item()


def mean_error(output, target):
    return torch.mean(output - target).item()