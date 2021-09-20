from typing import Union, List

import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image


_default_inverse_transform = torchvision.transforms.Compose([torchvision.transforms.Normalize((-0.3252/0.0265, -0.3283/0.0241, -0.3407/0.0252), (1/0.0265, 1/0.0241, 1/0.0252)), 
                                                            torchvision.transforms.ToPILImage()])

class MemoryDataset(Dataset):
    def __init__(self, 
                 inputs: list, 
                 targets: list, 
                 transform: callable=None, 
                 target_transform: list=None):
        
        if isinstance(inputs[0], torch.Tensor):
            inputs_ = [_default_inverse_transform(img) for img in inputs]
            inputs = inputs_
            
        self._inputs = inputs
        self._targets = targets
        self.transform = transform
        self.target_transform = target_transform
    
    @property
    def inputs(self):
        return self._inputs
    
    @property
    def targets(self):
        return self._targets
    
    def append(self, 
               inputs: Union[torch.Tensor, List[torch.Tensor]], 
               targets: Union[torch.Tensor, List[torch.Tensor]]):
        
        if isinstance(inputs, torch.Tensor) or isinstance(targets, torch.Tensor):
            inputs = [x for x in inputs]
            targets = [dict(label=y) for y in targets]
            
        # invers transform
        if isinstance(inputs[0], torch.Tensor):
            inputs_ = [_default_inverse_transform(img) for img in inputs]
            inputs = inputs_
            
        self._inputs += inputs
        self._targets += targets
    
    def get_labels(self):
        return [int(target['label']) for target in self.targets]
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        inputs = self._inputs[index]
        targets = self._targets[index]
        
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        
        return inputs, targets