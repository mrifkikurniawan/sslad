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