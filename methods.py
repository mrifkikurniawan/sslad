from easydict import EasyDict as edict

import torch.nn as nn
from avalanche.training.plugins import EWCPlugin

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
        self._model.fc = MLP(2048, 7)
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
        self._model.fc = MLP(2048, 7)
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
        return EWCPlugin(ewc_lambda=0.4, decay_factor=0.1, mode="separate")