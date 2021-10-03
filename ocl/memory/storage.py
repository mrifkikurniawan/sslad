from collections import defaultdict
from typing import Dict, List, Sequence, Union
from copy import deepcopy
import types
from tqdm import tqdm
from PIL import Image
import PIL

from randaugment import RandAugment
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchvision

from ocl.utils import *
from ocl.memory import MemoryDataset


_default_inverse_transform = torchvision.transforms.Compose([torchvision.transforms.Normalize((-0.3252/0.0265, -0.3283/0.0241, -0.3407/0.0252), (1/0.0265, 1/0.0241, 1/0.0252)), 
                                                            torchvision.transforms.ToPILImage()])

class OnlineCLStorage(object):
    def __init__(self,
                 transform: callable=None,
                 online_sampler: callable=None,
                 periodic_sampler: callable=None,
                 mem_size: int=1000,
                 target_layer: str=None,
                 embedding_dims: int=2048):
        super().__init__()
        
        self.inputs = list()
        self.targets = list()
        self._mem_size = mem_size
        self.transform = transform
        self.online_sampler = online_sampler
        self.periodic_sampler = periodic_sampler
        self.target_layer = target_layer
        self.embedding_dims = embedding_dims

    @property
    def current_capacity(self):
        return len(self.inputs)
    
    @property
    def mem_size(self):
        return self._mem_size
    
    @property
    def dataset(self):
        if self.transform is None:
            return MemoryDataset(inputs=self.inputs, targets=self.targets)
        else:
            return MemoryDataset(inputs=self.inputs, targets=self.targets, transform=self.transform)
            
    def periodic_update_memory(self, x: torch.Tensor, y: Dict[str, torch.Tensor], model: nn.Module, num_samples: int,  **kwargs):
        
        # append new batch datapoints into dataset
        # for sampling strategy
        if not 'feature' in y:
            batch_size = x.shape[0] 
            y['feature'] = torch.Tensor(batch_size, self.embedding_dims)
        self.dataset.append(inputs=x, targets=y)
        
        # sampling periodically
        inputs_samples, targets_samples = self.periodic_sampler(dataset=self.dataset, model=model, 
                                                                num_samples=num_samples, target_layer=self.target_layer, 
                                                                **kwargs)
        
        # assigne new samples to memory
        self.inputs = inputs_samples
        self.targets = targets_samples
        
        assert self.current_capacity <= self.mem_size, \
            f"Memory size is over capacity, max size is {self.mem_size} while current capacity is {self.current_capacity}"

    def online_update_memory(self, x: torch.Tensor, y: Dict[str, torch.Tensor], model: nn.Module, num_samples, **kwargs):
        """ Update memory with new experience. """
        remain_capacity = self.mem_size - self.current_capacity
        if num_samples > remain_capacity:
            num_samples = remain_capacity
        
        if self.current_capacity == self.mem_size:
            print(f"Current memory capacity is maximum, remove some within periodic_update_memory")
            pass
        elif self.current_capacity <= self.mem_size:
            inputs_samples, targets_samples = self.online_sampler(x=x, y=y, model=model, 
                                                                  num_samples=num_samples, 
                                                                  target_layer=self.target_layer, **kwargs)
            
            # transform inverse to original pil image if inputs is normalized tensor
            if isinstance(inputs_samples[0], torch.Tensor):
                inputs_samples_ = [_default_inverse_transform(img) for img in inputs_samples]
                inputs_samples = inputs_samples_
            
            self.inputs += inputs_samples
            self.targets += targets_samples
        else:
            raise ValueError(f"Memory size is over capacity")
        
        assert self.current_capacity <= self.mem_size, \
            f"Memory size is over capacity, max size is {self.mem_size} while current capacity is {self.current_capacity}"