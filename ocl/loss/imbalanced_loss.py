from typing import Optional
import numpy as np

from ocl.utils import get_batch_distribution, create_instance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
    

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
        Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight of label smooth.
    """
    def __init__(self, 
                 epsilon: float=0.1,
                 num_classes: int=7,
                 reduction: str="mean"):
        super().__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
        self.reduction = getattr(torch, reduction)
        
    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        targets = F.one_hot(targets, num_classes=self.num_classes)
        targets = targets.to(inputs.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = self.reduction((- targets * log_probs).mean(0))
        return loss



class CrossEntropyLabelAwareSmooth(nn.Module):
    """Cross entropy loss with label-aware smoothing regularizer.
    Reference:
        Zhong et al. Improving Calibration for Long-Tailed Recognition. CVPR 2021. https://arxiv.org/abs/2104.00466
    For more details of label-aware smoothing, you can see Section 3.2 in the above paper.
    Args:
        shape (str): the manner of how to get the params of label-aware smoothing.
        smooth_head (float): the largest  label smoothing factor
        smooth_tail (float): the smallest label smoothing factor
    """
    def __init__(self, 
                 num_classes: int=7,
                 smooth_head: float=0.4,
                 smooth_tail: float=0.1,
                 shape: str='concave'):
        super().__init__()

        self.shape = shape
        self.smooth_head  = torch.tensor(smooth_head)
        self.smooth_tail = torch.tensor(smooth_tail)
        self.num_classes = num_classes

    def get_smoothness(self, distribution: torch.Tensor) -> torch.Tensor:
        n_1 = torch.max(distribution)
        n_K = torch.min(distribution)
        if self.shape == 'concave':
            smooth = self.smooth_tail + (self.smooth_head - self.smooth_tail) * torch.sin((distribution - n_K) * torch.tensor(np.pi) / (2 * (n_1 - n_K)))
        elif self.shape == 'linear':
            smooth = self.smooth_tail + (self.smooth_head - self.smooth_tail) * (distribution - n_K) / (n_1 - n_K)
        elif self.shape == 'convex':
            smooth = self.smooth_head + (self.smooth_head - self.smooth_tail) * torch.sin(1.5 * torch.tensor(np.pi) + (distribution - n_K) * torch.tensor(np.pi) 
                                                                                          / (2 * (n_1 - n_K)))
        else:
            raise AttributeError  
        
        return smooth

    def forward(self, inputs, targets, **kwargs):
        
        # set smoothness level
        distribution = get_batch_distribution(targets, self.num_classes)
        smooth = self.get_smoothness(distribution).type_as(inputs)
        
        # measure loss
        smoothing = smooth[targets]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        
        return loss.mean()