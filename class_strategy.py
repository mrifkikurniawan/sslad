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

from utils import create_instance, cutmix_data

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
                 cut_mix: bool=True,
                 fast_dev_run: bool=False):
        super(ClassStrategyPlugin).__init__()
        
        sampler_to_module = {"rm": RMSampler(),
                             "random_split": random_split,
                             "uncertainty": UncertaintySampler()}
        
        self.mem_size = mem_size
        self.sampler = sampler_to_module[sampler]
        self.cut_mix = cut_mix
        self.ext_mem = dict()
        self.mem_transform = transforms.Compose([eval(transform) for transform in memory_transforms])
        self.storage_policy = MyStoragePolicty(self.ext_mem, self.mem_transform, self.sampler, self.mem_size)
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
        self.do_cutmix = self.cut_mix and np.random.rand(1) < 0.5
        if self.do_cutmix:
            strategy.mb_x, strategy.mb_y, labels_b, lambd = cutmix_data(x=strategy.mb_x, y=strategy.mb_y, alpha=1.0)
            self.cutmix_out = dict(labels_b=labels_b,
                                   lambd=lambd)

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        if self.do_cutmix:
            lambd = self.cutmix_out['lambd']
            labels_b = self.cutmix_out['labels_b']
            strategy.loss *= lambd
            strategy.loss += (1 - lambd) * strategy.criterion(strategy.mb_output, labels_b)

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
        uncertain_samples = sample_df.sort_values(by="uncertainty", ascending=False)
        selected_samples_indices = uncertain_samples.index[0:num_samples].tolist()

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
        


class UncertaintySampler(object):
    def __init__(self, 
                 augmentation: str="vr_randaug",
                 batch_size: int=32,
                 num_workers: int=32,
                 num_classes: int=7):
        
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.scoring = entropy
    
    def __call__(self, dataset: Union[AvalancheDataset, Subset], num_samples: int, model: nn.Module) -> Subset:
        uncertainy_score_per_sample = self._montecarlo(dataset, model)
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
        uncertain_samples = sample_df.sort_values(by="uncertainty", ascending=False)
        selected_samples_indices = uncertain_samples.index[0:num_samples].tolist()

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
            self._aggregate(sample, n_transforms)  
            
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
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        
        model.eval()
        with torch.no_grad():
            for _, mbatch in tqdm(enumerate(dataloader), desc=f"measure uncertainty"):
                x, y, _ = mbatch
                x = x.type_as(next(model.parameters()))
                logit = model(x)
                logit = logit.detach().cpu()

                for i, cert_value in enumerate(logit):
                    uncertainty_value = self.scoring(cert_value)
                    uncertainty_scores.append(uncertainty_value)
                    labels.append(y[i].item())

        if isinstance(dataset, AvalancheDataset):
            dataset._dataset._dataset.transform = original_transform
        elif isinstance(dataset, Subset):
            dataset.dataset._dataset._dataset.transform = original_transform
            
        return uncertainty_scores, labels

    def _aggregate(self, sample, cand_length):
        uncertainty_aggregate = torch.zeros(cand_length)
        for i in range(cand_length):
            uncertainty_aggregate[i] = sample[f"uncert_{str(i)}"]
        sample["uncertainty"] = torch.mean(uncertainty_aggregate)
        

def margin_conf(logit: torch.Tensor):
    proba_dist = nn.functional.softmax(logit, dim=0)
    sorted_proba_dist, _ = torch.sort(proba_dist, descending=True)
    most_conf = sorted_proba_dist[0]
    next_most_conf = sorted_proba_dist[1]
    uncertainty_score = 1 - (most_conf - next_most_conf)
    
    return uncertainty_score

def least_conf(logit: torch.Tensor):
    proba_dist = nn.functional.softmax(logit, dim=0)
    n_class = len(proba_dist)
    sorted_proba_dist, _ = torch.sort(proba_dist, descending=True)
    most_conf = sorted_proba_dist[0]
    uncertainty_score = (1 - most_conf) * n_class / (n_class - 1)
    
    return uncertainty_score

def entropy(logit: torch.Tensor):
    proba_dist = nn.functional.softmax(logit, dim=0)
    log_probs = torch.log(proba_dist)
    uncertainty_score = (proba_dist * -log_probs).sum()
    
    return uncertainty_score