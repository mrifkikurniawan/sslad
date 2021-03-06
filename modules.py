from collections import defaultdict
from typing import Dict, List, Union, Dict
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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.nn import functional as F

from avalanche.benchmarks.utils import  AvalancheDataset
from avalanche.training.strategies import BaseStrategy

from utils import *
from dataset import MemoryDataset, _default_inverse_transform
import scoring



class OnlineCLStorage(object):
    def __init__(self,
                 transform: callable=None,
                 online_sampler: callable=None,
                 periodic_sampler: callable=None,
                 mem_size: int=1000,
                 target_layer: str=None,
                 embedding_dims: int=2048):
        """[Episodic memory class to store the samples and execute online and periodic sampling.]

        Args:
            transform (callable, optional): [memory sample augmentations]. Defaults to None.
            online_sampler (callable, optional): [online sampler module]. Defaults to None.
            periodic_sampler (callable, optional): [periodic sampler module]. Defaults to None.
            mem_size (int, optional): [maximum capacity of memory]. Defaults to 1000.
            target_layer (str, optional): [target layer to get embeddings]. Defaults to None.
            embedding_dims (int, optional): [embeddings dimensions]. Defaults to 2048.
        """
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
        """[Periodic memory update function]

        Args:
            x (torch.Tensor): [input images. Dimensions: (batch_size, channels, height, width)]
            y (Dict[str, torch.Tensor]): [targets that containing multiple targets, e.g.: labels and logits]
            model (nn.Module): [model to get embeddings]
            num_samples (int): [number of samples to be selected]
        """        
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
        """[online memory update function]

        Args:
            x (torch.Tensor): [input images. Dimensions: (batch_size, channels, height, width)]
            y (Dict[str, torch.Tensor]): [targets that containing multiple targets, e.g.: labels and logits]
            model (nn.Module): [model to get embeddings]
            num_samples (int): [number of samples to be selected]
        """
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
        


class RMSampler(object):
    def __init__(self, 
                 augmentation: str="vr_randaug",
                 batch_size: int=32,
                 num_workers: int=32):
        """[Raimbow memory sampler mechanism adopted from original Rainbow Memory algorithm: https://github.com/clovaai/rainbow-memory]

        Args:
            augmentation (str, optional): [augmentation for monte-carlo]. Defaults to "vr_randaug".
            batch_size (int, optional): [batch size for pass forward]. Defaults to 32.
            num_workers (int, optional): [number of workers for dataloader]. Defaults to 32.
        """        
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.handlers = list()
    
    def __call__(self, dataset: Dataset, num_samples: int, model: nn.Module, target_layer: str) -> Image:
        if num_samples > len(dataset):
            num_samples = len(dataset)

        
        # register forward hook to get features
        self.target_layer = target_layer
        self._register_forward_hook(model)
        
        uncertainy_score_per_sample, outputs = self._montecarlo(dataset, model)
        sample_df = pd.DataFrame(uncertainy_score_per_sample)
        
        selected_samples_indices = self._select_indices(sample_df=sample_df, num_samples=num_samples)
        assert len(selected_samples_indices) == num_samples
        
        # storage the tensor samples to list
        selected_images = [dataset.inputs[idx] for idx in selected_samples_indices]
        selected_targets = [dataset.targets[idx] for idx in selected_samples_indices]
        logits = outputs['logits']
        features = outputs['features']
        for i, idx in enumerate(selected_samples_indices):
            selected_targets[i]['logit'] = logits[idx]
            selected_targets[i]['feature'] = features[idx]
        
        # remove hook
        self.remove_hook()

        
        return selected_images, selected_targets
    
    def _register_forward_hook(self, model: nn.Module):
        self.features = dict()
        def get_features(key):
            def forward_hook(module, input, output):
                self.features[key] = output#.detach()
            return forward_hook
        
        # add forward hook 
        for name, module in model.named_modules():
            if self.target_layer == name:
                self.handlers.append(module.register_forward_hook(get_features(key=name)))

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
            uncertainty_scores_per_augment[uncert_name], labels, outputs = self._compute_uncert(dataset, _tr, model=model)
        
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
            
        return uncertainy_score_per_sample, outputs
        
    def _compute_uncert(self, dataset: Dataset, transforms: List[callable], model: nn.Module) -> List[torch.Tensor]:
        batch_size = self.batch_size
        original_dataset_transform = deepcopy(dataset.transform)
        dataset.transform = transforms

        uncertainty_scores = list()
        labels = list()
        logits = list()
        features = list()
        outputs = dict()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        
        model.eval()
        with torch.no_grad():
            for _, mbatch in tqdm(enumerate(dataloader), desc=f"measure uncertainty"):
                try:
                    x, y, _ = mbatch
                except:
                    x, y = mbatch
                x = x.type_as(next(model.parameters()))
                logit = model(x)
                logit = logit.detach().cpu()

                for i, cert_value in enumerate(logit):
                    uncertainty_value = 1 - cert_value
                    uncertainty_scores.append(uncertainty_value)
                    labels.append(y['label'][i].item())
                    logits.append(cert_value)
                    features.append(self.features[self.target_layer][i].detach().cpu().squeeze())


        # return back the original transform
        dataset.transform = original_dataset_transform
        outputs['logits'] = logits
        outputs['features'] = features    
        return uncertainty_scores, labels, outputs

    def _variance_ratio(self, sample, cand_length):
        vote_counter = torch.zeros(sample["uncert_0"].size(0))
        for i in range(cand_length):
            top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item()
        
    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

class UncertaintySampler(object):
    def __init__(self, 
                 num_workers: int=32,
                 scoring_method: str="entropy",
                 negative_mining: bool=True):
        """[Sampler based on uncertainty scores]

        Args:
            num_workers (int, optional): [number of worker for dataloader]. Defaults to 32.
            scoring_method (str, optional): [uncertainty scoring method]. Defaults to "entropy".
            negative_mining (bool, optional): [negative mining flag]. Defaults to True.
        """        
        self.num_workers = num_workers
        self.scoring = getattr(scoring, scoring_method)
        self.negative_mining = negative_mining
        self.handlers = list()
    
    def __call__(self, x: torch.Tensor, y: Dict[str, torch.Tensor], model: nn.Module, num_samples: int, target_layer: str) -> List[torch.Tensor]:
        len_inputs = x.shape[0]
        if num_samples > len_inputs:
            num_samples = len_inputs

        # register forward hook to get features
        self.target_layer = target_layer
        self._register_forward_hook(model)
 
        labels = y['label']
        samples_scores, model_outputs = self._compute_score(x, labels, model)        
        selected_samples_indices = self._select_indices(samples_scores, num_samples=num_samples)
        
        assert len(selected_samples_indices) == num_samples
        selected_images = x[selected_samples_indices]
        
        # convert batch tensor to list of tensor
        selected_images = [x.cpu() for x in selected_images]
        selected_targets = [dict(label=y['label'][i].cpu(), 
                                 logit=model_outputs['logits'][i].cpu(), 
                                 feature=model_outputs['features'][i].cpu()) for i in selected_samples_indices]
        
        # remove hook
        self.remove_hook()

        
        return selected_images, selected_targets

    def _register_forward_hook(self, model: nn.Module):
        self.features = dict()
        def get_features(key):
            def forward_hook(module, input, output):
                self.features[key] = output#.detach()
            return forward_hook
        
        # add forward hook 
        for name, module in model.named_modules():
            if self.target_layer == name:
                self.handlers.append(module.register_forward_hook(get_features(key=name)))

    def _select_indices(self, samples_scores: torch.Tensor, num_samples: int) -> torch.Tensor:
        selected_samples_indices = torch.topk(samples_scores, k=num_samples, dim=0, largest=False)[1]

        return selected_samples_indices        

    def _compute_score(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> List[Dict]:
        
        # inference to get uncertainty scores
        # given latest updated model
        model.eval()
        logits = list()
        features = list()
        outputs = dict()
        
        with torch.no_grad():
            logit = model(x)
            proba_dist = nn.functional.softmax(logit, dim=1)

            # computing sampling scores based on uncertainty 
            # and negative mining if necessary
            samples_scores = self.scoring(proba_dist)
            if self.negative_mining:
                negative_scores = scoring.negative_scoring(proba_dist, y)
                samples_scores += negative_scores
            
            for i in range(x.shape[0]):
                logits.append(logit[i].detach())
                features.append(self.features[self.target_layer][i].detach().squeeze())
        
        outputs['logits'] = logits
        outputs['features'] = features
        return samples_scores, outputs        

                
    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove() 

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
    



class SoftmaxT(nn.Module):

    def __init__(self, temperature: Union[float, int, torch.Tensor]) -> None:
        """[Normalized softmax with temperature]

        Args:
            temperature (Union[float, int, torch.Tensor]): [temperature hparam] 
        """
        super(SoftmaxT, self).__init__()

        if isinstance(temperature, float) or isinstance(temperature, int):
            temperature = torch.tensor([temperature])
        self.temperature = temperature

    def forward(self, inputs: torch.Tensor, act_f: str) -> torch.Tensor:
        temperature = self.temperature.type_as(inputs)
        out = inputs/temperature

        if act_f == 'softmax':
            out = F.softmax(out, dim=1)
        elif act_f == 'log_softmax':
            out = F.log_softmax(out, dim=1)
        return out    



class ForwardTransfer(object):
    def __init__(self, 
                 patience: int=0,
                 weight: int=0.5,
                 scheduler: Union[edict, dict]=None
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
    
    
    
class MetricLearner(object):
    def __init__(self, losses: List[dict]):
        """[Deep metric learning module to measure metric learning similarity]

        Args:
            losses (List[dict]): [list of loss functions]
        """        
        # create loss instances
        self.loss_modules: List[dict] = list()
        for loss in losses:
            loss = edict(loss)
            loss_module = edict(module=create_instance(loss),
                                weight=torch.tensor(loss.weight))
            self.loss_modules.append(loss_module)
            
    def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """[pass forward call]

        Args:
            embeddings (torch.Tensor): [embeddings inputs. Dimensions: (batch_size, embedding_size)]
            labels (torch.Tensor): [labels targets. Dimensions: (batch_size,)]

        Returns:
            [torch.Tensor]: [loss value output]
        """
        loss = torch.tensor(0.0).type_as(embeddings)
        for i in range(len(self.loss_modules)):
            loss += self.loss_modules[i].module(embeddings, labels) * (self.loss_modules[i].weight).type_as(loss)
        return loss

