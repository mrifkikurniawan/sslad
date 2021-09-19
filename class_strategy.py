from collections import defaultdict
from typing import Dict, List, Sequence, Union
from copy import deepcopy
import types
from tqdm import tqdm
from easydict import EasyDict as edict

from randaugment import RandAugment
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Subset
import torchvision.transforms as transforms
from torchsampler import ImbalancedDatasetSampler

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheDataset
from avalanche.training.plugins import StoragePolicy
from avalanche.training.strategies import BaseStrategy

from utils import create_instance, cutmix_data
from modules import *

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
                 online_sampler: dict=None,
                 periodic_sampler: dict=None,
                 mem_size: int = 1000, 
                 memory_transforms: List[Dict] = None,
                 sweep_memory_every_n_iter: int=1000, 
                 memory_sweep_default_size: int=500,
                 num_samples_per_batch: int=5,
                 cut_mix: bool=True,
                 lr_scheduler: torch.optim.lr_scheduler=None,
                 temperature: float=0.5,
                 loss_weights: dict=None,
                 softlabels_patience: int=1000):
        super(ClassStrategyPlugin).__init__()
        
        self.mem_size = mem_size
        self.current_itaration = 0
        self.sweep_memory_every_n_iter = sweep_memory_every_n_iter
        self.memory_sweep_default_size = memory_sweep_default_size
        self.num_samples_per_batch = num_samples_per_batch
        
        # lr scheduler
        self.lr_scheduler = lr_scheduler
        
        # episodic memory
        self.mem_transform = transforms.Compose([eval(transform) for transform in memory_transforms])
        self.online_sampler = create_instance(online_sampler)
        self.periodic_sampler = create_instance(periodic_sampler)
        self.storage = OnlineCLStorage(self.mem_transform, self.online_sampler, 
                                       self.periodic_sampler, self.mem_size)
        self.memory_dataloader = None
        
        # augmentations
        self.cut_mix = cut_mix

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_training_exp(self, strategy: 'BaseStrategy', 
                            num_workers: int = 16, 
                            shuffle: bool = True,
                            **kwargs):
        pass

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                        **kwargs):
        pass

    def after_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                       **kwargs):
        pass

    def before_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        
        # having some memory, then join the current batch with the small batch of memory            
        if self.storage.current_capacity == self.memory_sweep_default_size:
            self.memory_dataloader = iter(DataLoader(self.storage.dataset, 
                                                     batch_size=6, 
                                                     shuffle=False,
                                                     num_workers=self.online_sampler.num_workers,
                                                     sampler=ImbalancedDatasetSampler(self.storage.dataset)))
            
            
    def before_forward(self, strategy: 'BaseStrategy', **kwargs):
        if self.memory_dataloader:
            try:
                x_memory, y_memory = self.memory_dataloader.next()
                x_memory = x_memory.type_as(strategy.mb_x)
                y_memory = y_memory.type_as(strategy.mb_y)
                strategy.mbatch[0] = torch.cat([strategy.mb_x, x_memory], dim=0)
                strategy.mbatch[1] = torch.cat([strategy.mb_y, y_memory], dim=0)
            except:
                pass
        
        # cut mix augmentation if necessary
        if self.cut_mix:
            self.do_cutmix = self.cut_mix and np.random.rand(1) < 0.5
            if self.do_cutmix:
                strategy.mbatch[0], strategy.mbatch[1], labels_b, lambd = cutmix_data(x=strategy.mb_x, y=strategy.mb_y, alpha=0.25)
                self.cutmix_out = dict(labels_b=labels_b,
                                       lambd=lambd)

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        
        # cut mix augmentation if necessary
        if self.cut_mix:
            if self.do_cutmix:
                lambd = self.cutmix_out['lambd']
                labels_b = self.cutmix_out['labels_b']
                strategy.loss *= lambd
                strategy.loss += (1 - lambd) * strategy._criterion(strategy.mb_output, labels_b)

    def after_backward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_update(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_update(self, strategy: 'BaseStrategy', **kwargs):
        
        # get model, current batch images and targets 
        model = strategy.model
        x, y = strategy.mb_x, strategy.mb_y
        
        # online episodic memory update
        self.storage.online_update_memory(x, y, model, self.num_samples_per_batch)
        
        # periodic episodic memory update
        if self.current_itaration > 0:
            if self.current_itaration % self.sweep_memory_every_n_iter == 0 or \
                self.storage.current_capacity == self.mem_size:
                    x, y = x.detach().cpu(), y.detach().cpu()
                    print(f"----- Periodic episodic memory update, sweep up some memories -----")
                    print(f"Current training steps: {self.current_itaration}")
                    print(f"Current storage capacity: {self.storage.current_capacity}")
                    num_samples = self.memory_sweep_default_size
                    self.storage.periodic_update_memory(x, y, model, num_samples)
            
        
        # learning rate scheduler step()
        if self.lr_scheduler:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(strategy.loss.item())
            else:
                self.lr_scheduler.step()
        
        self.current_itaration += 1

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_dataset_adaptation(self, strategy: 'BaseStrategy',
        if self.current_itaration >= self.softlabels_patience:
            self.softlabels_learning = True
            print("------- Starting softlabels learning -------")
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