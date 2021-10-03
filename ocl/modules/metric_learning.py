from typing import List
from easydict import EasyDict as edict

import torch
from ocl.utils import create_instance

class MetricLearner(object):
    def __init__(self, losses: List[dict]):
        
        # create loss instances
        self.loss_modules: List[dict] = list()
        for loss in losses:
            loss = edict(loss)
            loss_module = edict(module=create_instance(loss),
                                weight=torch.tensor(loss.weight))
            self.loss_modules.append(loss_module)
            
    def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        loss = torch.tensor(0.0).type_as(embeddings)
        for i in range(len(self.loss_modules)):
            loss += self.loss_modules[i].module(embeddings, labels) * (self.loss_modules[i].weight).type_as(loss)
        return loss