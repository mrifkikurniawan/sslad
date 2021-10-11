from typing import Optional
import numpy as np

from utils import get_batch_distribution, create_instance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['KLDivLoss', 'FocalLoss']


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
    
    
    
class FocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: Optional[float] = 2.0,
                 reduction: Optional[str] = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: Optional[float] = gamma
        self.reduction: Optional[str] = reduction
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor, 
            weight: torch.Tensor=None) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 2:
            raise ValueError("Invalid input shape, we expect BxD. Got: {}"
                             .format(input.shape))
        if not input.shape[0] == target.shape[0]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1) + self.eps

        # create the labels one hot tensor
        target_one_hot = F.one_hot(target, num_classes=input.shape[1])

        # compute the actual focal loss
        fl_weight = torch.pow(1. - input_soft, self.gamma)
        focal = -self.alpha * fl_weight * torch.log(input_soft)
        loss_tmp = target_one_hot * focal
        
        # weighting the loss
        if weight is not None: 
            assert loss_tmp.shape[1] == weight.shape[0]
            loss_tmp = torch.mul(loss_tmp, weight)
        
        loss_tmp = torch.sum(loss_tmp, dim=1)

        loss = -1
        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        return loss
    

class MultipleLosses(torch.nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.is_dict = isinstance(losses, dict)
        self.losses = (
            torch.nn.ModuleDict(losses) if self.is_dict else torch.nn.ModuleList(losses)
        )

        if weights is not None:
            self.assertions_if_not_none(weights, match_all_keys=True)
            self.weights = weights
        else:
            self.weights = (
                {k: 1 for k in self.losses.keys()}
                if self.is_dict
                else [1] * len(losses)
            )

    def forward(self, inputs, labels):
        total_loss = 0
        iterable = self.losses.items() if self.is_dict else enumerate(self.losses)
        for i, loss_func in iterable:
            total_loss += (
                loss_func(inputs, labels) * self.weights[i]
            )
        return total_loss

    def assertions_if_not_none(self, x, match_all_keys):
        if x is not None:
            if self.is_dict:
                assert isinstance(x, dict)
                if match_all_keys:
                    assert sorted(list(x.keys())) == sorted(list(self.losses.keys()))
                else:
                    assert all(k in self.losses.keys() for k in x.keys())
            else:
                assert len(x) == len(self.losses)
                
                

class CostSenstiveLoss(nn.Module):
    def __init__(self, 
                 loss: dict=None,
                 num_classes: int=7, 
                 **kwargs: dict):
        super().__init__()
        
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.dataset_distribution = torch.zeros(self.num_classes)
        self.num_iter = 0
        self.epsilon = 1e-6
        self.loss = create_instance(loss)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        loss = torch.tensor(0.0).type_as(inputs)
        distribution = get_batch_distribution(targets, self.num_classes) + self.epsilon
        weight = (torch.min(distribution[1:]) / distribution).type_as(inputs)
        loss += self.loss(inputs, targets, weight=weight, **self.kwargs)
        
        self.num_iter += 1
        self.dataset_distribution += distribution
        
        return loss
    

class ClassBalancedLoss(nn.Module): 
    def __init__(self, 
                 loss: dict=None,
                 beta: float=0.9,
                 num_classes: int=7, 
                 **kwargs):
        super().__init__()
        
        self.beta = torch.tensor(beta) 
        self.loss = create_instance(loss)
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.dataset_distribution = torch.zeros(self.num_classes)
        self.num_iter = 0
        self.epsilon = 1e-6
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        loss = torch.tensor(0.0).type_as(inputs)
        distribution = get_batch_distribution(targets, self.num_classes) + self.epsilon
        weight = ((1.0 - self.beta) / (1.0 - self.beta**distribution)).type_as(inputs)
        loss += self.loss(inputs, targets, weight=weight, **self.kwargs)
        
        self.num_iter += 1
        self.dataset_distribution += distribution
        
        return loss

