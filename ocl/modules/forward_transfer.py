from typing import Union
from easydict import EasyDict as edict
import torch
import torch.nn as nn

from ocl.utils import clone_module



class ForwardTransfer(object):
    def __init__(self, 
                 patience: int=0,
                 weight: int=0.5,
                 ):
        self.patience = patience
        self.step_counter = 0
        self.weight = weight
        self.execute_transfer = False
    
    def update_temporary_model(self, model: nn.Module):
        self.temporary_model = clone_module(model)
    
    def step(self):
        if self.step_counter == self.patience:
            self.execute_transfer = True
            print(f"Start Forward Transfer on step {self.step_counter}")
            print(f"Weight: {self.weight}")
                
        self.step_counter += 1
        
    def criterion(self, model: nn.Module):
        if self.execute_transfer:
            loss = torch.tensor(0.0).to(next(model.parameters()).device)
            
            for current_param, temporary_param in zip(model.named_parameters(), self.temporary_model.named_parameters()):
                assert current_param[0] == temporary_param[0], "Module name should be similar"
                loss += torch.norm(current_param[1] - temporary_param[1], p=2)
            
            loss = self.weight * loss
        else:
            loss = torch.tensor(0.0).to(next(model.parameters()).device)
        
        return loss