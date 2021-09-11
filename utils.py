from easydict import EasyDict as edict
from importlib import import_module
from typing import Union
import numpy as np

import torch



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
    
# from https://github.com/drimpossible/GDumb/blob/74a5e814afd89b19476cd0ea4287d09a7df3c7a8/src/utils.py#L102:5
def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2