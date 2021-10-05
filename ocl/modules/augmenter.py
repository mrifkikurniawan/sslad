import numpy as np
import torch
import torch.nn as nn
from avalanche.training.strategies import BaseStrategy

from ocl.utils import get_batch_distribution


class MixUp(object):
    def __init__(self, 
                 alpha: float=None):
    
        self.alpha = alpha
        self.distribution = torch.distributions.beta.Beta(self.alpha, self.alpha)
        
    def _augment(self, inputs: torch.Tensor, targets: torch.Tensor):
        # sample lambda from beta distribution
        self.lambd = self.distribution.sample().type_as(inputs)
        
        # random permutation
        idx = torch.randperm(inputs.size(0))
        image_a, image_b = inputs, inputs[idx]
        self.label_a, self.label_b = targets, targets[idx]
        
        # mix original order images with permuted order images
        self.mixed_image = (self.lambd * image_a) + ((1 - self.lambd) * image_b)
    
    def step(self, strategy: BaseStrategy):
        
        # augment image
        self._augment(strategy.mb_x, strategy.mb_y)
        
        # pass forward
        strategy.optimizer.zero_grad()
        logits = strategy.model(self.mixed_image)
        
        # weighting the loss
        strategy.loss = torch.tensor(0.0).type_as(strategy.loss)
        strategy.loss +=  self.lambd * strategy.criterion(logits, self.label_a) +  (1 - self.lambd) * strategy.criterion(logits, self.label_b)
        
        # backward
        strategy.loss.backward()
        strategy.optimizer.step()



class ReMix(object):
    def __init__(self, 
                 alpha: float=None,
                 tau: float=None,
                 kappa: float=None):
    
        self.alpha = alpha
        self.distribution = torch.distributions.beta.Beta(self.alpha, self.alpha)
        self.tau = tau
        self.kappa = kappa
        
    def _augment(self, inputs: torch.Tensor, targets: torch.Tensor):
        # sample lambda from beta distribution
        self.lambd = self.distribution.sample().type_as(inputs)
        
        # random permutation
        idx = torch.randperm(inputs.size(0))
        image_a, image_b = inputs, inputs[idx]
        self.label_a, self.label_b = targets, targets[idx]
        
        # mix original order images with permuted order images
        self.mixed_image = (self.lambd * image_a) + ((1 - self.lambd) * image_b)
    
    def _get_loss_weights(self, class_distribution: torch.Tensor):
        weights = torch.empty(self.mixed_image.shape[0]).fill_(self.lambd).float()
        n_i, n_j = class_distribution[self.label_a], class_distribution[self.label_b].float()
        if self.lambd < self.tau:
            weights[n_i / n_j >= self.kappa] = 0
        if 1 - self.lambd < self.tau:
            weights[(n_i * self.kappa) / n_j <= 1] = 1

        return weights
    
    def step(self, strategy: BaseStrategy):        
        # augment image
        self._augment(strategy.mb_x, strategy.mb_y)
        
        # get points-wise weights
        class_distribution = get_batch_distribution(strategy.mb_y, num_classes=7)
        weights = self._get_loss_weights(class_distribution)
        
        # pass forward
        strategy.optimizer.zero_grad()
        logits = strategy.model(self.mixed_image)
        
        # weighting the loss
        strategy.loss = torch.tensor(0.0).type_as(strategy.loss)
        
        strategy.criterion.weight = weights
        strategy.loss +=  strategy.criterion(logits, self.label_a)
        strategy.criterion.weight = 1 - weights
        strategy.loss += strategy.criterion(logits, self.label_b)
        
        # backward
        strategy.loss.backward()
        strategy.optimizer.step()
        
        # reset criterion weights
        strategy.criterion.weight = None



class CutMix(object):
    def __init__(self, 
                 alpha: float=None):
    
        self.alpha = alpha
        self.distribution = torch.distributions.beta.Beta(self.alpha, self.alpha)
        
    def _augment(self, inputs: torch.Tensor, targets: torch.Tensor):
        # sample lambda from beta distribution
        self.lambd = self.distribution.sample().type_as(inputs)
        
        # random permutation
        idx = torch.randperm(inputs.size(0))
        
        # random crop and mix
        self.label_a, self.label_b = targets, targets[idx]
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(inputs.size(), self.lambd)
        self.mixed_image = inputs.detach().clone()
        self.mixed_image[:, :, bbx1:bbx2, bby1:bby2] = self.mixed_image[idx, :, bbx1:bbx2, bby1:bby2]
        
        # adjust lambda to exactly match pixel ratio
        self.lambd = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (self.mixed_image.size()[-1] * self.mixed_image.size()[-2]))
        
    def step(self, strategy: BaseStrategy):
        
        # augment image
        self._augment(strategy.mb_x, strategy.mb_y)
        
        # pass forward
        strategy.optimizer.zero_grad()
        logits = strategy.model(self.mixed_image)
        
        # weighting the loss
        strategy.loss = torch.tensor(0.0).type_as(strategy.loss)
        strategy.loss +=  self.lambd * strategy.criterion(logits, self.label_a) +  (1 - self.lambd) * strategy.criterion(logits, self.label_b)
        
        # backward
        strategy.loss.backward()
        strategy.optimizer.step()
    
    def _rand_bbox(size: torch.Tensor, lam: torch.Tensor):
        W = torch.tensor(size[2])
        H = torch.tensor(size[3])
        cut_rat = torch.sqrt(1.0 - lam)
        cut_w = (W * cut_rat).int()
        cut_h = (H * cut_rat).int()

        # uniform
        cx = torch.randint(0, W, (1, ))
        cy = torch.randint(0, H, (1, ))

        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2