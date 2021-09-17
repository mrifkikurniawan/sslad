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
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
import torchvision.transforms as transforms

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheDataset
from avalanche.training.strategies import BaseStrategy

from utils import create_instance, cutmix_data
from dataset import MemoryDataset, _default_inverse_transform
import scoring



class OnlineCLStorage(object):
    def __init__(self,
                 transform: callable=None,
                 online_sampler: callable=None,
                 periodic_sampler: callable=None,
                 mem_size: int=1000):
        super().__init__()
        
        self.inputs = list()
        self.targets = list()
        self._mem_size = mem_size
        self.transform = transform
        self.online_sampler = online_sampler
        self.periodic_sampler = periodic_sampler

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
            
    def periodic_update_memory(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module, num_samples: int,  **kwargs):
        
        # append new batch datapoints into dataset
        # for sampling strategy
        self.dataset.append(inputs=x, targets=y)
        
        # sampling periodically
        inputs_samples, targets_samples = self.periodic_sampler(dataset=self.dataset, model=model, num_samples=num_samples, **kwargs)
        
        # assigne new samples to memory
        self.inputs = inputs_samples
        self.targets = targets_samples
        
        assert self.current_capacity <= self.mem_size, \
            f"Memory size is over capacity, max size is {self.mem_size} while current capacity is {self.current_capacity}"

    def online_update_memory(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module, **kwargs):
        """ Update memory with new experience. """
        num_samples = self.mem_size - self.current_capacity
        
        if self.current_capacity == self.mem_size:
            print(f"Current memory capacity is maximum, remove some within periodic_update_memory")
            pass
        elif self.current_capacity <= self.mem_size:
            inputs_samples, targets_samples = self.online_sampler(x=x, y=y, model=model, 
                                                                  num_samples=num_samples, **kwargs)
            
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
        


class RMSampler(object):
    def __init__(self, 
                 augmentation: str="vr_randaug",
                 batch_size: int=32,
                 num_workers: int=32):
        
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def __call__(self, dataset: Dataset, num_samples: int, model: nn.Module) -> List[PIL.Image]:
        
        uncertainy_score_per_sample = self._montecarlo(dataset, model)
        sample_df = pd.DataFrame(uncertainy_score_per_sample)
        
        selected_samples_indices = self._select_indices(sample_df=sample_df, num_samples=num_samples)
        assert len(selected_samples_indices) == num_samples
        
        # storage the tensor samples to list
        selected_images = [dataset.inputs[idx] for idx in selected_samples_indices]
        selected_targets = [dataset.targets[idx] for idx in selected_samples_indices]
        
        return selected_images, selected_targets
       

    def _select_indices(self, sample_df: pd.DataFrame, num_samples: int) -> List[int]:
        uncertain_samples = sample_df.sort_values(by="uncertainty", ascending=False)
        selected_samples_indices = uncertain_samples.index[0:num_samples].tolist()

        return selected_samples_indices
    
    def _init_augmentation(self) -> List[callable]:
        transform_cands = list()
        if self.augmentation == "vr_randaug":
            for _ in range(12):
                transform_cands.append(RandAugment())
        
        return transform_cands         

    def _montecarlo(self, dataset: Dataset, model: nn.Module) -> List[Dict]:
        # initialize list of augmenters
        augmentation_module = self._init_augmentation()
        n_transforms = len(augmentation_module)
        uncertainty_scores_per_augment = dict()
        uncertainy_score_per_sample = list()
        
        # inference on each single augmenter to obtain uncertainty score per augmenter
        for idx, tr in enumerate(augmentation_module):
            _tr = transforms.Compose([tr] + [transforms.ToTensor(), transforms.Normalize((0.3252, 0.3283, 0.3407), (0.0265, 0.0241, 0.0252))])
            uncert_name = f"uncert_{str(idx)}"
            uncertainty_scores_per_augment[uncert_name], labels = self._compute_uncert(dataset, _tr, model=model)
        
        len_samples = len(dataset)
        for i in range(len_samples):
            score = dict(label=labels[i])
            for uncert_name in uncertainty_scores_per_augment:
                unc_scores_sample_i = uncertainty_scores_per_augment[uncert_name][i]
                score[uncert_name] = unc_scores_sample_i
            uncertainy_score_per_sample.append(score)
        
        for i in range(len_samples):
            sample = uncertainy_score_per_sample[i]
            self._variance_ratio(sample, n_transforms)  
            
        return uncertainy_score_per_sample
        
    def _compute_uncert(self, dataset: Dataset, transforms: List[callable], model: nn.Module) -> List[torch.Tensor]:
        batch_size = self.batch_size
        original_dataset_transform = deepcopy(dataset.transform)
        dataset.transform = transforms

        uncertainty_scores = list()
        labels = list()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        
        model.eval()
        with torch.no_grad():
            for _, mbatch in tqdm(enumerate(dataloader), desc=f"measure uncertainty"):
                x, y, _ = mbatch
                x = x.type_as(next(model.parameters()))
                logit = model(x)
                logit = logit.detach().cpu()

                for i, cert_value in enumerate(logit):
                    uncertainty_value = 1 - cert_value
                    uncertainty_scores.append(uncertainty_value)
                    labels.append(y[i].item())

        # return back the original transform
        dataset.transform = original_dataset_transform
            
        return uncertainty_scores, labels

    def _variance_ratio(self, sample, cand_length):
        vote_counter = torch.zeros(sample["uncert_0"].size(0))
        for i in range(cand_length):
            top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item()
        


class UncertaintySampler(object):
    def __init__(self, 
                 num_workers: int=32,
                 scoring_method: str="entropy",
                 negative_mining: bool=True):
        
        self.num_workers = num_workers
        self.scoring = getattr(scoring, scoring_method)
        self.negative_mining = negative_mining
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module, num_samples: int,) -> List[torch.Tensor]:
        samples_scores = self._compute_score(x, y, model)        
        selected_samples_indices = self._select_indices(samples_scores, num_samples=num_samples)
        
        assert len(selected_samples_indices) == num_samples
        selected_images, selected_targets = x[selected_samples_indices], y[selected_samples_indices]
        
        # convert batch tensor to list of tensor
        selected_images = [x for x in selected_images]
        selected_targets = [y for y in selected_targets]
        
        return selected_images, selected_targets

    def _select_indices(self, samples_scores: torch.Tensor, num_samples: int) -> torch.Tensor:
        selected_samples_indices = torch.topk(samples_scores, k=num_samples, dim=0, largest=False)[1]

        return selected_samples_indices        

    def _compute_score(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> List[Dict]:
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, 10, shuffle=False, num_workers=self.num_workers)
        
        # inference to get uncertainty scores
        # given latest updated model
        model.eval()
        with torch.no_grad():
            for _, mbatch in enumerate(dataloader):
                x, y = mbatch
                x = x.type_as(next(model.parameters()))
                logit = model(x)
                proba_dist = nn.functional.softmax(logit, dim=1)

                # computing sampling scores based on uncertainty 
                # and negative mining if necessary
                samples_scores = self.scoring(proba_dist)
                if self.negative_mining:
                    negative_scores = negative_scoring(proba_dist, y)
                    samples_scores += negative_scores
        
        return samples_scores        
                
        

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
    
    