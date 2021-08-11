from easydict import EasyDict as edict
from importlib import import_module
from typing import Union



def create_instance(module_cfg: Union[dict, edict], **kwargs):
    if isinstance(module_cfg, dict):
        module_cfg = edict(module_cfg)
    
    module = module_cfg.module
    method = module_cfg.method
    module_args = module_cfg.get("args", None)
    
    if module_args is not None:
        module_args.update(kwargs)
    
    print(f"Initializing {module}.{method}")
    module = import_module(module)
    module = getattr(module, method)
    
    if module_args != None:
        instance = module(**module_args)
    else:
        instance = module()
        
    return instance


def seed_everything(seed_value):
  import os
  os.environ['PYTHONHASHSEED'] = str(seed_value)
  import random 
  random.seed(seed_value) # Python
  import numpy as np
  np.random.seed(seed_value) # cpu vars
  import torch
  torch.manual_seed(seed_value) # cpu  vars
  
  if torch.cuda.is_available(): 
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) # gpu vars
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False