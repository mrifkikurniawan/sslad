from typing import Optional, Dict, List, Union
import numpy as np

from utils import get_batch_distribution, create_instance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['KLDivLoss', 'FocalLoss']


class KLDivLoss(nn.Module):
    def __init__(self, temperature: float, **kwargs):
        """[KL Divergence Loss]

        Args:
            temperature (float): [temperature for normalized softmax]
        """
        super(KLDivLoss, self).__init__()
        self.temperature = torch.Tensor([temperature])
        
    def forward(self, logits: torch.Tensor, y: torch.Tensor):
        """[forward pass]

        Args:
            logits (torch.Tensor): [logits from model. Dimensions: (batch_size, num_classes)]] 
            y (torch.Tensor): [target labels. Dimensions: (batch_size, num_classes)]

        Returns:
            [torch.Tensor]: [computed loss value]
        """
        y = y.type_as(logits)
        self.temperature = self.temperature.type_as(logits)
        
        y_hat = F.log_softmax(logits/self.temperature, dim=1)
        loss = F.kl_div(y_hat, y, reduction='batchmean') * (self.temperature**2)
        return loss.squeeze()
    
    
    
class FocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: Optional[float] = 2.0,
                 reduction: Optional[str] = 'none') -> None:
        """[Focal Loss Class. Hacked from: https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py]

        Args:
            alpha (float): [alpha hyperparameter for focal loss]
            gamma (Optional[float], optional): [gamma hyperparams]. Defaults to 2.0.
            reduction (Optional[str], optional): [reduction type]. Defaults to 'none'.
        """
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
        """[focal loss forward pass]

        Args:
            input (torch.Tensor): [input tensor. Dims: (batch_size, num_classes)]
            target (torch.Tensor): [target tensor. Dims: (batch_size, num_classes)]
            weight (torch.Tensor, optional): [weigths for each class. Dims: (num_classes)]. Defaults to None.

        Returns:
            torch.Tensor: [computed loss value]
        """
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
    def __init__(self, losses: Union[Dict, List], weights: Union[Dict, List]=None):
        """[Multiple Loss Wrapper Module. Hacked from: https://github.com/KevinMusgrave/pytorch-metric-learning/blob/6bfa880b8d2acafb7c6d52041d2bb14ed41aee76/src/pytorch_metric_learning/losses/base_metric_loss_function.py#L62]

        Args:
            losses (Union[Dict, List]): [losses instance in dictionary or list]
            weights (Union[Dict, List], optional): [weights associated to each loss]. Defaults to None.
        """
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
        """[forward pass]

        Args:
            inputs ([torch.Tensor]): [inputs to loss module]
            labels ([torch.Tensor]): [targets to loss module]

        Returns:
            [torch.Tensor]: [loss value]
        """
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
        """[Cost Sensitive Loss Class.]

        Args:
            loss (dict, optional): [loss configuration]. Defaults to None.
            num_classes (int, optional): [number of classes]. Defaults to 7.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.dataset_distribution = torch.zeros(self.num_classes)
        self.num_iter = 0
        self.epsilon = 1e-6
        self.loss = create_instance(loss)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """[forward pass]

        Args:
            inputs (torch.Tensor): [inputs]
            targets (torch.Tensor): [targets]

        Returns:
            [torch.Tensor]: [loss value]
        """
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
        """[Class Balanced Loss Module]

        Args:
            loss (dict, optional): [loss function configuration]. Defaults to None.
            beta (float, optional): [beta hyperparam]. Defaults to 0.9.
            num_classes (int, optional): [number of classes]. Defaults to 7.
        """
        super().__init__()
        
        self.beta = torch.tensor(beta) 
        self.loss = create_instance(loss)
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.dataset_distribution = torch.zeros(self.num_classes)
        self.num_iter = 0
        self.epsilon = 1e-6
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """[pass forward]

        Args:
            inputs (torch.Tensor): [inputs feed to loss.]
            targets (torch.Tensor): [targets feed to loss.]

        Returns:
            [torch.Tensor]: [loss value]
        """
        loss = torch.tensor(0.0).type_as(inputs)
        distribution = get_batch_distribution(targets, self.num_classes) + self.epsilon
        weight = ((1.0 - self.beta) / (1.0 - self.beta**distribution)).type_as(inputs)
        loss += self.loss(inputs, targets, weight=weight, **self.kwargs)
        
        self.num_iter += 1
        self.dataset_distribution += distribution
        
        return loss

