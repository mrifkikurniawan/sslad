from typing import Dict, List, Union
from easydict import EasyDict as edict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies import BaseStrategy

from ocl.memory import OnlineCLStorage
from ocl.utils import create_instance, cutmix_data
from ocl.modules import *
from ocl.loss import *

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
                 cut_mix: Union[bool, dict]=False,
                 ep_memory_batch_size: int=6,
                 lr_scheduler: torch.optim.lr_scheduler=None,
                 logger: object=None,
                 target_layer: str=None,
                 metric_learning: dict=None,
                 embedding_dims: int=None,
                 memory_dataloader_sampler: dict=None, 
                 finetune_head: dict=None,
                 model: nn.Module=None,
                 softlabels_trainer: object=None):
        super(ClassStrategyPlugin).__init__()
        
        self.mem_size = mem_size
        self.current_itaration = 0
        self.sweep_memory_every_n_iter = sweep_memory_every_n_iter
        self.memory_sweep_default_size = memory_sweep_default_size
        self.num_samples_per_batch = num_samples_per_batch
        self.target_layer = target_layer
        self.embedding_dims = embedding_dims
        
        # lr scheduler
        self.lr_scheduler = lr_scheduler
        
        # episodic memory
        self.mem_transform = transforms.Compose([eval(transform) for transform in memory_transforms])
        self.online_sampler = create_instance(online_sampler)
        self.periodic_sampler = create_instance(periodic_sampler)
        self.storage = OnlineCLStorage(self.mem_transform, self.online_sampler, 
                                       self.periodic_sampler, self.mem_size, target_layer,
                                       self.embedding_dims)
        self.memory_dataloader = False
        self.ep_memory_batch_size = ep_memory_batch_size
        self.sampler = create_instance(memory_dataloader_sampler)
        
        # augmentations
        self.cut_mix = cut_mix
        if self.cut_mix:
            self.cut_mix = edict(self.cut_mix)

        # -------- Soft Labels --------
        self.softlabels_trainer = softlabels_trainer
        if self.softlabels_trainer:
            self.softlabels_trainer = create_instance(self.softlabels_trainer)
                
        # logger
        self.logger = logger
        
        # metric learning
        self.metric_learning = metric_learning
        if self.metric_learning:
            self.handlers = list()
            self.metric_learner = create_instance(self.metric_learning)
            
        # finetune head
        self.finetune_head = finetune_head
        if self.finetune_head:
            self.finetune_head = create_instance(finetune_head, model=model)

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_training_exp(self, strategy: 'BaseStrategy', 
                            num_workers: int = 16, 
                            shuffle: bool = True,
                            **kwargs):
        if self.metric_learning:
            self._register_forward_hook(strategy.model)

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
                                                     batch_size=self.ep_memory_batch_size, 
                                                     shuffle=False,
                                                     num_workers=self.online_sampler.num_workers,
                                                     sampler=self.sampler(self.storage.dataset)))
            
            if self.finetune_head:
                self.finetune_head.load_dataset(self.storage.dataset)
                self.finetune_head.prepare_dataloader()
            
            
    def before_forward(self, strategy: 'BaseStrategy', **kwargs):
        if self.memory_dataloader:
            try:
                x_memory, self.y_memory = self.memory_dataloader.next()
                x_memory = x_memory.type_as(strategy.mb_x)
                y_memory = self.y_memory['label'].type_as(strategy.mb_y)
                strategy.mbatch[0] = torch.cat([strategy.mb_x, x_memory], dim=0)
                strategy.mbatch[1] = torch.cat([strategy.mb_y, y_memory], dim=0)
            except:
                pass
        
        # cut mix augmentation if necessary
        if self.cut_mix:
            self.do_cutmix = self.cut_mix and np.random.rand(1) < self.cut_mix.probability
            if self.do_cutmix:
                strategy.mbatch[0], strategy.mbatch[1], labels_b, lambd = cutmix_data(x=strategy.mb_x, y=strategy.mb_y, alpha=self.cut_mix.alpha)
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

        # soft labels learning
        if self.softlabels_trainer.train and self.memory_dataloader:
            memory_batch_size = self.y_memory['logit'].shape[0]
            logits = strategy.mb_output[-memory_batch_size:]
            softlabels_loss = self.softlabels_trainer.fit(inputs=logits, targets=self.y_memory['logit'].type_as(logits))
            strategy.loss *= torch.tensor(self.softlabels_trainer.ce_weights).type_as(strategy.loss)
            strategy.loss += softlabels_loss
        
        # step softlabels trainer
        if self.softlabels_trainer:
            self.softlabels_trainer.step()
         
        # metric learning
        if self.metric_learning:
            loss = self.metric_learner(embeddings=self.embeddings[self.target_layer].squeeze(), labels=strategy.mb_y)
            strategy.loss += loss.type_as(strategy.loss)

    def after_backward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_update(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_update(self, strategy: 'BaseStrategy', **kwargs):
        
        # finetune model head
        if self.finetune_head:
            self.finetune_head.step()
        
        # get model, current batch images and targets 
        model = strategy.model
        x, y = strategy.mb_x.detach(), strategy.mb_y.detach()
        logits = strategy.mb_output.detach()
        
        # online episodic memory update
        self.storage.online_update_memory(x, dict(label=y, logit=logits), model, self.num_samples_per_batch)
        
        # periodic episodic memory update
        if self.current_itaration > 0:
            if self.current_itaration % self.sweep_memory_every_n_iter == 0 or \
                self.storage.current_capacity == self.mem_size:
                    x, y, logits = x.cpu(), y.cpu(), logits.cpu()
                    print(f"----- Periodic episodic memory update, sweep up some memories -----")
                    print(f"Current training steps: {self.current_itaration}")
                    print(f"Current storage capacity: {self.storage.current_capacity}")
                    num_samples = self.memory_sweep_default_size
                    self.storage.periodic_update_memory(x, dict(label=y, logit=logits), model, num_samples)
            
        
        # learning rate scheduler step()
        if self.lr_scheduler:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(strategy.loss.item())
            else:
                self.lr_scheduler.step()
        
        self.current_itaration += 1
            
        # logging
        self.logger.log({"train/loss": strategy.loss.item()})

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        if self.metric_learning:
            self.remove_hook()

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
    
    def _register_forward_hook(self, model: nn.Module):
        self.embeddings = dict()
        def get_features(key):
            def forward_hook(module, input, output):
                self.embeddings[key] = output
            return forward_hook
        
        # add forward hook 
        for name, module in model.named_modules():
            if self.target_layer == name:
                self.handlers.append(module.register_forward_hook(get_features(key=name)))
                
    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove() 