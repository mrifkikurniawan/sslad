from typing import Dict
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from avalanche.benchmarks.utils import AvalancheDataset
from utils import *



class Prototypes(object):
    def __init__(self, 
                 num_k: int=10,
                 ):
        
        self.num_k_per_class = num_k
        self._embedding = dict()
        
    def update(self, dataset: AvalancheDataset, model: nn.Module):
        self._embedding = self._compute_prototypes(dataset, model)
    
    def _compute_prototypes(self, memory_dataset: AvalancheDataset, model: nn.Module):
        prototypes_per_class = dict()
        features_per_class = defaultdict(list)
        
        # get all episodic memory features embedding from the current model
        dataloader = DataLoader(memory_dataset, batch_size=32, shuffle=False)
        for i, data in enumerate(dataloader):
            x, targets = data
            features = model(x.type_as(next(model.parameters())))
            features = features.detach().cpu()
            
            for j, target in enumerate(targets):
                features_per_class[int(target)] = features[j]
        
        # select prototypes per class
        for class_id in features_per_class:
            features = features_per_class[class_id] # (n_samples, n_features)
            features = torch.stack(features, dim=0)
            
            # calculate the mean of class features
            class_mean_features = torch.mean(features, dim=0) # (n_features)
            class_mean_features = class_mean_features.reshape(1, class_mean_features.shape[0]) # (1, n_features)
            
            # measure distance using l2
            distance = torch.cdist(features, class_mean_features, p=2.0) # (n_samples, 1)
            _, sorted_indices = torch.sort(distance.reshape(len(distance)), descending=False)
            
            # select top-k nearest prototypes to mean
            prototypes_per_class[class_id] = torch.index_select(features, dim=0, index=sorted_indices)
        
        return prototypes_per_class
        
    @property
    def embedding(self) -> Dict[str, torch.Tensor]:
        return self._embedding