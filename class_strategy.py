from typing import Dict, List
from copy import deepcopy

from randaugment import RandAugment

import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheDataset
from avalanche.training.plugins import StoragePolicy
from avalanche.training.strategies import BaseStrategy

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
                 fast_dev_run: bool=False):
        super(ClassStrategyPlugin).__init__()
        
        self.mem_size = mem_size
        self.ext_mem = dict()
        self.mem_transform = transforms.Compose([eval(transform) for transform in memory_transforms])
        self.storage_policy = MyStoragePolicty(self.ext_mem, self.mem_transform, self.mem_size)
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
                 mem_size: int=1000):
        super().__init__(ext_mem, mem_size)
        
        self.transform = transform
        
    def subsample_single(self, dataset: AvalancheDataset, num_samples: int, model: nn.Module):
        removed_els = len(dataset) - num_samples
        if removed_els > 0:
            data, _ = random_split(dataset, [num_samples, removed_els])
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
        current_dataset = self.subsample_single(current_dataset, new_group_size, model)
        self.ext_mem[strategy.training_exp_counter + 1] = current_dataset
        
        # buffer size should always equal self.mem_size
        len_tot = sum(len(el) for el in self.ext_mem.values())
        assert len_tot == self.mem_size