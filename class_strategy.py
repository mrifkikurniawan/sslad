from typing import Dict, List, Sequence, Union
from copy import deepcopy
import types
from tqdm import tqdm

from randaugment import RandAugment
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Subset
import torchvision.transforms as transforms

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheDataset
from avalanche.training.plugins import StoragePolicy
from avalanche.training.strategies import BaseStrategy

from utils import create_instance

"""
A strategy pulgin can change the strategy parameters before and after 
every important fucntion. The structure is shown belown. For instance, self.before_train_forward() and
self.after_train_forward() will be called before and after the main forward loop of the strategy.

The training loop is organized as follows::
        **Training loop**
            train
                train_exp  # for each experience
                    adapt_train_dataset
                    train_dataset_adaptation
                    make_train_dataloader
                    train_epoch  # for each epoch
                        train_iteration # for each minibatch
                            forward
                            backward
                            model update

        **Evaluation loop**
        The evaluation loop is organized as follows::
            eval
                eval_exp  # for each experience
                    adapt_eval_dataset
                    eval_dataset_adaptation
                    make_eval_dataloader
                    eval_epoch  # for each epoch
                        eval_iteration # for each minibatch
                            forward
                            backward
                            model update
"""

class ClassStrategyPlugin(StrategyPlugin):

    def __init__(self, 
                 mem_size: int = 1000, 
                 memory_transforms: List[Dict] = None,
                 sampler: str="random_split",
                 fast_dev_run: bool=False):
        super(ClassStrategyPlugin).__init__()
        
        self.mem_size = mem_size
        if sampler == "rm":
            self.sampler = RMSampler()
        elif sampler == "random_split": 
            self.sampler = sampler
        self.ext_mem = dict()
        self.mem_transform = transforms.Compose([eval(transform) for transform in memory_transforms])
        self.storage_policy = MyStoragePolicty(self.ext_mem, self.mem_transform, self.mem_size, sampler=self.sampler)
        self.fast_dev_run = fast_dev_run

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_training_exp(self, strategy: 'BaseStrategy', 
                            num_workers: int = 16, 
                            shuffle: bool = True,
                            **kwargs):
        if self.fast_dev_run:
            strategy.adapted_dataset = random_split(strategy.adapted_dataset, lengths=10)
        
        if len(self.ext_mem) == 0:
            return
        
        print("create dataloader of episodic memory and current data stream")
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            AvalancheConcatDataset(self.ext_mem.values()),
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                        **kwargs):
        pass

    def after_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                       **kwargs):
        pass

    def before_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_backward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_update(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_update(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        self.storage_policy(strategy, **kwargs)

    def after_training(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_dataset_adaptation(self, strategy: 'BaseStrategy',
                                       **kwargs):
        pass

    def after_eval_dataset_adaptation(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_exp(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval_exp(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass
    
    
class MyStoragePolicty(StoragePolicy):
    def __init__(self,
                 ext_mem: dict,
                 transform: callable=None,
                 sampler: callable=random_split,
                 mem_size: int=1000):
        super().__init__(ext_mem, mem_size)
        
        self.transform = transform
        self.sampler = sampler
        
    def subsample_single(self, dataset: AvalancheDataset, num_samples: int, **kwargs):
        removed_els = len(dataset) - num_samples
        if removed_els > 0:
            if isinstance(self.sampler, types.FunctionType):
                if self.sampler.__name__ == "random_split":
                    data, _ = self.sampler(dataset, [num_samples, removed_els], **kwargs)
            else:
                data = self.sampler(dataset, num_samples, **kwargs)
        return data
            
    def subsample_all_groups(self, new_size, **kwargs):
        """ Subsample all groups equally to match total buffer size
        `new_size`. """
        groups = list(self.ext_mem.keys())
        if len(groups) == 0:
            return  # buffer is empty.

        num_groups = len(groups)
        group_size = new_size // num_groups
        last_group_size = group_size + (new_size % num_groups)

        for g in groups[:-1]:
            self.ext_mem[g] = self.subsample_single(self.ext_mem[g], group_size, **kwargs)
        # last group may be bigger
        last = self.ext_mem[groups[-1]]
        self.ext_mem[groups[-1]] = self.subsample_single(last, last_group_size,**kwargs)
    
    def __call__(self, strategy: BaseStrategy, **kwargs):
        num_exps = strategy.training_exp_counter + 1
        current_dataset = strategy.experience.dataset
        model = strategy.model
        
        # replace with new transformation with augmentations inside
        current_dataset._dataset._dataset.transform = self.transform
        
        past_group_size = self.mem_size // num_exps
        new_group_size = past_group_size + (self.mem_size % num_exps)
        
        self.subsample_all_groups(past_group_size * (num_exps - 1), model=model)
        current_dataset = self.subsample_single(current_dataset, new_group_size, model=model)
        self.ext_mem[strategy.training_exp_counter + 1] = current_dataset
        
        # buffer size should always equal self.mem_size
        len_tot = sum(len(el) for el in self.ext_mem.values())
        assert len_tot == self.mem_size
        

class RMSampler(object):
    def __init__(self, 
                 augmentation: str="vr_randaug",
                 batch_size: int=32,
                 num_workers: int=32,
                 num_classes: int=7):
        
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
    
    def __call__(self, dataset: Union[AvalancheDataset, Subset], num_samples: int, model: nn.Module) -> Subset:
        uncertainy_score_per_sample = self._montecarlo(dataset, model)
        selected_samples_indices = list()
        sample_df = pd.DataFrame(uncertainy_score_per_sample)
        
        selected_samples_indices = self._select_indices(sample_df=sample_df, num_samples=num_samples)
        
        print("num samples:", num_samples)
        print("Len selected samples:", len(selected_samples_indices))
        assert len(selected_samples_indices) == num_samples
       
        if isinstance(dataset, Subset):
            dataset.indices = np.array(dataset.indices).take(selected_samples_indices).tolist()
            return dataset
        elif isinstance(dataset, AvalancheDataset):
            return Subset(dataset, indices=selected_samples_indices)

    def _select_indices(self, sample_df: pd.DataFrame, num_samples: int) -> List[int]:
        index_tracker = {i:0 for i in range(self.num_classes)}
        selected_samples_indices = list()
        mem_per_cls = num_samples // self.num_classes
        num_residual = num_samples % self.num_classes
        occupation_counter = num_samples
        
        for i in range(self.num_classes):
            cls_df = sample_df[sample_df["label"] == i]
            len_cls_df = len(cls_df)
            if len_cls_df <= mem_per_cls:
                indices = cls_df.index.tolist()
                selected_samples_indices += indices
                occupation_counter -= len(indices)
                index_tracker[i] = len(indices)
                
                # recounting the memory per class
                mem_per_cls = occupation_counter // self.num_classes - (i+1)
                num_residual = occupation_counter % self.num_classes - (i+1)
            else:
                uncertain_samples = cls_df.sort_values(by="uncertainty", ascending=False)
                if mem_per_cls + num_residual <= len(uncertain_samples):
                    indices = uncertain_samples.index[0:mem_per_cls + num_residual].tolist()
                    num_residual = 0
                else:
                    indices = uncertain_samples.index[0:mem_per_cls].tolist()
                selected_samples_indices += indices
                occupation_counter -= len(indices)  
                index_tracker[i] = len(indices)          
        
        while len(selected_samples_indices) < num_samples:
            class_distribution = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                cls_df = sample_df[sample_df["label"] == i] 
                class_distribution[i] = len(cls_df)
            mem_per_cls = np.round(occupation_counter * class_distribution/class_distribution.sum()).astype(int)
            num_residual = occupation_counter - np.sum(mem_per_cls) 
            cls_max = np.argmax(class_distribution)
            mem_per_cls[cls_max] = mem_per_cls[cls_max] + num_residual
            assert np.sum(mem_per_cls) == occupation_counter, f"{np.sum(mem_per_cls)}, {occupation_counter}"
            
            for i in range(self.num_classes):
                cls_df = sample_df[sample_df["label"] == i]            
                last_idx = index_tracker[i]
                uncertain_samples = cls_df.sort_values(by="uncertainty", ascending=False).index[last_idx:].tolist()
                if len(uncertain_samples) == 0:
                    indices = cls_df.sort_values(by="uncertainty", ascending=False).index[0:mem_per_cls[i]].tolist()
                else:
                    indices = uncertain_samples[0:mem_per_cls[i]]
                selected_samples_indices += indices
                occupation_counter -= len(indices)

        return selected_samples_indices
    
    def _init_augmentation(self) -> List[callable]:
        transform_cands = list()
        if self.augmentation == "vr_randaug":
            for _ in range(12):
                transform_cands.append(RandAugment())
        
        return transform_cands         

    def _montecarlo(self, dataset: AvalancheDataset, model: nn.Module) -> List[Dict]:
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
        
    def _compute_uncert(self, dataset: AvalancheDataset, transforms: List[callable], model: nn.Module) -> List[torch.Tensor]:
        batch_size = self.batch_size
        if isinstance(dataset, AvalancheDataset):
            original_transform = dataset._dataset._dataset.transform
            dataset._dataset._dataset.transform = transforms
        elif isinstance(dataset, Subset):
            original_transform = dataset.dataset._dataset._dataset.transform
            dataset.dataset._dataset._dataset.transform = transforms

        uncertainty_scores = list()
        labels = list()
        infer_dataset._dataset._dataset.transform = transforms
        dataloader = DataLoader(infer_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        
        model.eval()
        with torch.no_grad():
            for _, mbatch in enumerate(dataloader):
                x, y, _ = mbatch
                x = x.type_as(model)
                logit = self.model(x)
                logit = logit.detach().cpu()

                for i, cert_value in enumerate(logit):
                    uncertainty_value = 1 - cert_value
                    uncertainty_scores.append(uncertainty_value)
                    labels.append(y[i].item())

        if isinstance(dataset, AvalancheDataset):
            dataset._dataset._dataset.transform = original_transform
        elif isinstance(dataset, Subset):
            dataset.dataset._dataset._dataset.transform = original_transform
            
        return uncertainty_scores, labels

    def _variance_ratio(self, sample, cand_length):
        vote_counter = torch.zeros(sample["uncert_0"].size(0))
        for i in range(cand_length):
            top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item()