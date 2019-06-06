import torch.nn.functional as F
import torch


def RMSE(output, target):
    return (F.mse_loss(output, target) ** 0.5).item()


def EVA(output, target):
    """Note: this does not work with batch size 1"""
    diff = output - target
    return (1 - diff.var() / target.var()).item()


def mean_error(output, target):
    """To see if the model tends to predict on one side"""
    return torch.mean(output - target).item()

def output_std(output, target):
    """To see if the output of the model isn't the same"""
    return output.std().item()