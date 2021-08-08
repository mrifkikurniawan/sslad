from easydict import Easydict as edict

from class_strategy import *
from utils import create_instance

class Naive(object):
    def __init__(self, 
                 model: edict,
                 optimizer: edict,
                 criterion: edict
                 ):
        
        self._model = create_instance(model)
        self._optimizer = getattr(optimizer.module, optimizer.method)(self._model.parameters(), **.optimizerargs)
        self._criterion = create_instance(criterion) 
    
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
        self._plugins = ClassStrategyPlugin()