import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, output, target):
        pt = F.softmax(output, dim=-1)

        logpt = F.log_softmax(output, dim=1)
        py = F.softmax(target, dim=1)
        klloss = nn.KLDivLoss(reduction='none')
        kl_loss = klloss(logpt, py)

        ce = self.alpha * kl_loss

        focal_term = (1 - pt) ** self.gamma

        focal_loss = focal_term * ce

        if self.size_average == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        return focal_loss

def soft_loss(outputs, targets, norm=None):
    log_softmax_outputs = F.log_softmax(outputs / 3.0, dim=1)
    softmax_targets = F.softmax(targets / 3.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()