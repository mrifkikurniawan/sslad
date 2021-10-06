from typing import List, Dict, Union

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from avalanche.training.strategies import BaseStrategy

from ocl.utils import create_instance
from ocl.memory import SSLDataset
from ocl.modules import SoftmaxT
from ocl.loss import KLDivLoss

class FinetuneHeadTrainer(object):
    def __init__(self,
                 criterion: dict=None,
                 optimizer: dict=None,
                 model: nn.Module=None,
                 lr_scheduler: dict=None,
                 layer_head: str="fc",
                 dataloader: dict=None,
                 dataloader_sampler: dict=None,
                 softlabels_trainer: dict=None) -> None:
        
        self.criterion = create_instance(criterion)
        self.model = model
        self.layer_head = layer_head
        self.model_head = getattr(self.model, self.layer_head)
        self.optimizer = create_instance(optimizer, params=self.model_head.parameters())
        self.dataset = None
        self.dataloader_cfg = dataloader
        self.dataloader_sampler_cfg = dataloader_sampler
        self.lr_scheduler = lr_scheduler
        self.softlabels_trainer = softlabels_trainer
        
        if self.lr_scheduler:
            self.lr_scheduler = create_instance(lr_scheduler, optimizer=self.optimizer)
        
        if self.softlabels_trainer:
            self.softlabels_trainer = create_instance(self.softlabels_trainer)
        
    def load_dataset(self, dataset: torch.utils.data.Dataset) -> None:
        self.dataset = dataset
        
    def prepare_dataloader(self) -> None:
        assert self.dataset is not None, "Dataset not loaded"
        
        # prepare dataloader sampler
        self.dataloader_sampler = create_instance(self.dataloader_sampler_cfg, dataset=self.dataset) if self.dataloader_sampler_cfg else None
        self.dataloader = create_instance(self.dataloader_cfg, 
                                          dataset=self.dataset,
                                          sampler=self.dataloader_sampler)
        self.dataset_iter = iter(self.dataloader)
        
    def step(self, **kwargs):
        if hasattr(self, 'dataloader'):
            # set default values for training
            self.optimizer.zero_grad()
            device = next(self.model.parameters()).device
            loss = torch.tensor(0.0).to(device)
            
            # pass forward
            try:
                x, y = self.dataset_iter.next()
            except StopIteration:
                # reinitialize dataloader
                self.dataset_iter = iter(self.dataloader)
                x, y = self.dataset_iter.next()
                
            x, y, y_logits = x.to(device), y['label'].to(device), y['logit'].to(device)
            logits = self.model(x, **kwargs)
            
            # softlabels training
            loss += self.criterion(logits, y)
            if self.softlabels_trainer:
                if self.softlabels_trainer.train:
                    loss *= torch.tensor(self.softlabels_trainer.ce_weights).type_as(loss)
                    loss += self.softlabels_trainer.fit(logits, y_logits)
                self.softlabels_trainer.step()
            
            # backward
            loss.backward()
            self.optimizer.step()
            
            # scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()
        else:
            self.softlabels_trainer.step()
            pass



class SoftLabelsLearningTrainer(object):
    def __init__(self, 
                 patience: int=None,
                 temperature: int=None,
                 loss_weights: Dict[str, List[float]]=None):
    
        self.patience = patience
        self.temperature = temperature
        self.loss_weights = loss_weights
        self.iteration = 0
        self.next_milestone = self.loss_weights['milestones'][0]
        self.kldiv_loss = KLDivLoss(temperature=self.temperature)
        self.softmaxt = SoftmaxT(temperature=self.temperature)
        self._train = False
        self.milestone_idx = 0
    
    def configure_weight(self):
        if self.iteration == self.next_milestone:
            self.milestone_idx = self.loss_weights['milestones'].index(self.next_milestone)
            self.current_milestone = self.next_milestone
            last_milestone = self.loss_weights['milestones'][-1]
            if self.current_milestone != last_milestone:
                self.next_milestone = self.loss_weights['milestones'][self.milestone_idx+1]
            print(f"Current milestone: {self.current_milestone}")
            print(f"CE Loss: {self.loss_weights['cross_entropy'][self.milestone_idx]}")
            print(f"KL Loss: {self.loss_weights['kl_divergence'][self.milestone_idx]}")
    
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        if self.train:
            self.configure_weight()
            loss = torch.tensor(0.0).type_as(predictions)
            softlabels = self.softmaxt(targets, act_f='softmax')
            loss += self.kldiv_loss(logits=predictions, y=softlabels)
            loss *= torch.tensor(self.loss_weights['kl_divergence'][self.milestone_idx]).type_as(loss)
        else:
            loss = torch.tensor(0.0).type_as(predictions)
        return loss
                
    def step(self):
        self.iteration += 1
        if self.iteration == self.patience:
            self._train = True
            print("------- Starting Softlabels Learning -------")
            
    @property
    def train(self):
        return self._train
    
    @property
    def ce_weights(self):
        return self.loss_weights['cross_entropy'][self.milestone_idx]



class SelfSupervisedTrainer(nn.Module):
    def __init__(self, 
                 model: nn.Module=None,
                 head: dict=None,
                 target_layer: str=None,
                 criterion: dict=None,
                 optimizer: dict=None,
                 lr_scheduler: dict=None,
                 train_individually: bool=False,
                 inputs_tranforms: str=None,
                 targets_tranforms: str=None,
                 train: bool=True,
                 num_workers: int=8,
                 feed_targets_to_model: bool=True
                 ):
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.inputs_tranforms = inputs_tranforms
        self.targets_tranforms = targets_tranforms
        self.num_workers = num_workers 
        self.feed_targets_to_model = feed_targets_to_model
        self.handlers = list()
        self._register_forward_hook()

        if self.train:
            print("------ Self-Supervised Learning ------")
            self.head = create_instance(head)
            self.criterion = create_instance(criterion)
            self.optimizer = create_instance(optimizer)
            self.lr_scheduler = create_instance(lr_scheduler)
            self.train_individually = train_individually
            self.train = train
        
    def fit(self, strategy: BaseStrategy) -> None:
        self.device = next(strategy.model.parameters()).device
        if self.train:
            inputs = strategy.mbatch[0]
            self.prepare_dataloader(inputs)
            self.forward()            
            loss = self.criterion(self.preds, self.y)           
            
            if self.train_individually:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                strategy.loss += loss    
        else:
            pass
    
    def prepare_dataloader(self, inputs: torch.Tensor) -> None:
        batch_size = inputs.shape[0]
        dataset = SSLDataset(inputs, self.inputs_tranforms, self.targets_tranforms)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        
    def forward(self) -> None:
        x, self.y = self.dataloader.next()
        if self.feed_targets_to_model:
            self.features_out = dict()
            self.logits_x = self.model(x.to(self.device))       
            self.preds = self.embeddings[self.target_layer]
            self.logits_y = self.model(self.y.to(self.device))
            self.y = self.embeddings[self.target_layer]
            
            assert self.preds != self.y
        
        else:
            self.logits_x = self.model(x.to(self.device))
            self.logits_y = None
            self.preds = self.embeddings[self.target_layer]
                
    def adapt(self, inputs: torch.Tensor) -> None:
        '''Using Test-time adaptation'''
        
        self.optimizer.zero_grad()
        self.model.train()
        self.prepare_dataloader(inputs)
        self.forward()
        loss = self.criterion(self.preds, self.y)
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        
    def _register_forward_hook(self) -> None:
        self.embeddings = dict()
        def get_features(key):
            def forward_hook(module, input, output):
                self.embeddings[key] = output
            return forward_hook
        
        # add forward hook 
        for name, module in self.model.named_modules():
            if self.target_layer == name:
                self.handlers.append(module.register_forward_hook(get_features(key=name)))
                
    def remove_hook(self) -> None:
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove() 