from easydict import EasyDict as edict

import torch.nn as nn
from avalanche.training.plugins import EWCPlugin, ReplayPlugin, SynapticIntelligencePlugin, AGEMPlugin, LwFPlugin, CoPEPlugin, CWRStarPlugin
import timm

from class_strategy import *
from utils import create_instance
from models import MLP

class CLStrategy(object):
    def __init__(self, 
                 model: edict,
                 optimizer: edict,
                 criterion: edict,
                 plugins: edict,
                 lr_scheduler: edict,
                 logger: edict
                 ):
        """[Our continal learning strategy method]

        Args:
            model (edict): [model configuration]
            optimizer (edict): [optimizer configuration]
            criterion (edict): [loss function configuration]
            plugins (edict): [plugins configuration]
            lr_scheduler (edict): [lr scheduler configuration]
            logger (edict): [logger configuration]
        """
        # model
        head_layer = model.head_layer
        embedding_dims = model.embedding_dims
        self._model = create_instance(deepcopy(model))
        classifier = nn.Linear(embedding_dims, 7, bias=False)
        setattr(self._model, head_layer, classifier)
        
        self._optimizer = create_instance(deepcopy(optimizer), params=self._model.parameters())
        self._lr_scheduler = create_instance(deepcopy(lr_scheduler), optimizer=self._optimizer)
        if criterion['method'] == "MultipleLosses":
            losses = [create_instance(loss) for loss in criterion['args']['losses']]
            self._criterion = create_instance(deepcopy(criterion), losses=losses)
        else:
            self._criterion = create_instance(deepcopy(criterion)) 
        self._logger = create_instance(logger,
                                  config={'model': model, 
                                          'optimizer': optimizer, 
                                          'criterion': criterion, 
                                          'lr_scheduler/method': lr_scheduler,
                                          'plugins': plugins,
                                        }
                                  )
        self._plugins = create_instance(plugins, lr_scheduler=self._lr_scheduler, logger=self._logger,
                                        embedding_dims=embedding_dims)
    
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
    
    @property
    def logger(self):
        return self._logger
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
                 selection_strategy: dict=None,
                 features_based: bool=False
                 ):
        
        self._model = create_instance(model)
        self._model.fc = nn.Linear(2048, 7, bias=False)
        self._optimizer = create_instance(optimizer, params=self._model.parameters())
        self._criterion = create_instance(criterion) 
        self._mem_size = mem_size
        
        # if features based exemplar strategy
        # feeding model and its layername for features extractor
        if features_based:
            self._selection_strategy = create_instance(selection_strategy, model=self._model) if not selection_strategy is None else None
        else:    
            self._selection_strategy = create_instance(selection_strategy) if not selection_strategy is None else None
        
        # create storage policy with selection strategy if pre-defined
        if self._selection_strategy:
            try:
                self._storage_policy = create_instance(storage_policy, selection_strategy=self._selection_strategy)
                self._storage_policy.ext_mem = dict()
                print(f"selection strategy: {self._selection_strategy}")
            except:
                self._storage_policy = create_instance(storage_policy)
                self._storage_policy.ext_mem = dict()
                print(f"Selection strategy: None")
        else:
            print(f"Selection strategy: None")
            self._storage_policy = create_instance(storage_policy)
            self._storage_policy.ext_mem = dict()
            
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
        return ReplayPlugin(self._mem_size, self._storage_policy)
    
    
class SynapticIntelligence(object):
    def __init__(self, 
                 model: edict,
                 optimizer: edict,
                 criterion: edict,
                 si_lambda: float,
                 excluded_parameters: str="fc"
                 ):
        
        self._model = create_instance(model)
        self._model.fc = nn.Linear(2048, 7, bias=False)
        self._optimizer = create_instance(optimizer, params=self._model.parameters())
        self._criterion = create_instance(criterion) 
        self._si_lambda = si_lambda
        self._excluded_parameters = excluded_parameters
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
        return SynapticIntelligencePlugin(si_lambda=self._si_lambda, 
                                          excluded_parameters=self._excluded_parameters)
        

class AGEM(object):
    def __init__(self, 
                 model: edict,
                 optimizer: edict,
                 criterion: edict,
                 **kwargs
                 ):
        
        self._model = create_instance(model)
        self._model.fc = nn.Linear(2048, 7, bias=False)
        self._optimizer = create_instance(optimizer, params=self._model.parameters())
        self._criterion = create_instance(criterion) 
        self._kwargs = kwargs
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
        return AGEMPlugin(**self._kwargs)
    
    
    
class LwF(object):
    def __init__(self, 
                 model: edict,
                 optimizer: edict,
                 criterion: edict,
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
        return LwFPlugin()
    
    
    
class CoPE(object):
    def __init__(self, 
                 model: edict,
                 optimizer: edict,
                 criterion: edict,
                 **kwargs
                 ):
        
        self._model = create_instance(model)
        self._model.fc = nn.Linear(2048, 7, bias=False)
        self._optimizer = create_instance(optimizer, params=self._model.parameters())
        self._criterion = create_instance(criterion) 
        self._kwargs = kwargs
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
        return CoPEPlugin(**self._kwargs)
    
    
class CWRStar(object):
    def __init__(self, 
                 model: edict,
                 optimizer: edict,
                 criterion: edict,
                 **kwargs
                 ):
        
        self._model = create_instance(model)
        self._model.fc = nn.Linear(2048, 7, bias=False)
        self._optimizer = create_instance(optimizer, params=self._model.parameters())
        self._criterion = create_instance(criterion) 
        self._kwargs = kwargs
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
        return CWRStarPlugin(**self._kwargs, model=self._model)