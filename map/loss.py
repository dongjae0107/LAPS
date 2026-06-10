import torch
import torch.nn as nn
import torch.nn.functional as F


def sdf_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target, reduction='mean')

def sdf_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target, reduction='mean')

def sdf_huber_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(pred, target, reduction='mean')