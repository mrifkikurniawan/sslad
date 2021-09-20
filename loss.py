import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['KLDivLoss']

class KLDivLoss(nn.Module):
    def __init__(self, temperature: float, **kwargs):
        super(KLDivLoss, self).__init__()
        self.temperature = torch.Tensor([temperature])
        
    def forward(self, logits: torch.Tensor, y: torch.Tensor):
        y = y.type_as(logits)
        self.temperature = self.temperature.type_as(logits)
        
        y_hat = F.log_softmax(logits/self.temperature, dim=1)
        loss = F.kl_div(y_hat, y, reduction='batchmean') * (self.temperature**2)
        return loss.squeeze()