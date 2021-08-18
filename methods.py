from easydict import EasyDict as edict

import torch.nn as nn
from avalanche.training.plugins import EWCPlugin, ReplayPlugin

from class_strategy import *
from utils import create_instance
from models import MLP

class NaiveFinetune(object):
    def __init__(self, 
                 model: edict,
                 optimizer: edict,
                 criterion: edict
                 ):
        
        self._model = create_instance(model)
        self._model.fc = nn.Linear(2048, 7, bias=False)
        self._optimizer = create_instance(optimizer, params=self._model.parameters())
        self._criterion = create_instance(criterion) 
        self._plugins = self.initialize_plugins()
    
    @property
    def model(self):
        return self._model
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def criterion(self):
        return self._criterion
    
    @property
    def plugins(self):
        return self._plugins
    
    def initialize_plugins(self):
        return ClassStrategyPlugin()
    
    
class EWC(object):
    def __init__(self, 
                 model: edict,
                 optimizer: edict,
                 criterion: edict
                 ):
        
        self._model = create_instance(model)
        self._model.fc = nn.Linear(2048, 7, bias=False)
        self._optimizer = create_instance(optimizer, params=self._model.parameters())
        self._criterion = create_instance(criterion) 
        self._plugins = self.initialize_plugins()
    
    @property
    def model(self):
        return self._model
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def criterion(self):
        return self._criterion
    
    @property
    def plugins(self):
        return self._plugins
    
    def initialize_plugins(self):
        return EWCPlugin(ewc_lambda=0.4, decay_factor=0.1, mode="online")


class Replay(object):
    def __init__(self, 
                 model: edict,
                 optimizer: edict,
                 criterion: edict,
                 mem_size: int,
                 storage_policy: dict,
                 selection_strategy: dict,
                 ):
        
        self._model = create_instance(model)
        self._model.fc = nn.Linear(2048, 7, bias=False)
        self._optimizer = create_instance(optimizer, params=self._model.parameters())
        self._criterion = create_instance(criterion) 
        self._plugins = self.initialize_plugins()
        self._mem_size = mem_size
        self._selection_strategy = create_instance(selection_strategy) if not selection_strategy is None else None
        
        # create storage policy with selection strategy if pre-defined
        if self._selection_strategy:
            try:
                self._storage_policy = create_instance(storage_policy, selection_strategy=self._selection_strategy)
                print(f"selection strategy: {self._selection_strategy}")
            except:
                self._storage_policy = create_instance(storage_policy)
                print(f"Selection strategy: None")
        else:
            print(f"Selection strategy: None")
            self._storage_policy = create_instance(storage_policy)
    
    @property
    def model(self):
        return self._model
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def criterion(self):
        return self._criterion
    
    @property
    def plugins(self):
        return self._plugins
    
    def initialize_plugins(self):
        return ReplayPlugin(self._mem_size, self._storage_policy)