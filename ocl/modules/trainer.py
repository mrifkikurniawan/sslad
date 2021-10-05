import torch.nn as nn
import torch

from ocl.utils import create_instance

class FinetuneHeadTrainer(object):
    def __init__(self,
                 criterion: dict=None,
                 optimizer: dict=None,
                 model: nn.Module=None,
                 scheduler: dict=None,
                 layer_head: str="fc",
                 dataloader: dict=None,
                 dataloader_sampler: dict=None) -> None:
        
        self.criterion = create_instance(criterion)
        self.model = model
        self.layer_head = layer_head
        self.model_head = getattr(self.model, self.layer_head)
        self.optimizer = create_instance(optimizer, params=self.model_head.parameters())
        self.dataset = None
        self.dataloader_cfg = dataloader
        self.dataloader_sampler_cfg = dataloader_sampler
        self.scheduler = scheduler
        
        if self.scheduler:
            self.scheduler = create_instance(scheduler)
        
    def load_dataset(self, dataset: torch.utils.data.Dataset) -> None:
        self.dataset = dataset
        
    def prepare_dataloader(self) -> None:
        assert self.dataset is not None, "Dataset not loaded"
        
        # prepare dataloader sampler
        self.dataloader_sampler = create_instance(self.dataloader_sampler_cfg, dataset=self.dataset) if self.dataloader_sampler_cfg else None
        self.dataloader = iter(create_instance(self.dataloader_cfg, 
                                               dataset=self.dataset,
                                               sampler=self.dataloader_sampler))
        
    def step(self, **kwargs):
        if hasattr(self, 'dataloader'):
            # set default values for training
            self.optimizer.zero_grad()
            device = next(self.model.parameters()).device
            loss = torch.tensor(0.0).to(device)
            
            # pass forward
            x, y = self.dataloader.next()
            x, y = x.to(device), y['label'].to(device)
            logits = self.model(x, **kwargs)
            loss += self.criterion(logits, y)
            
            # backward
            loss.backward()
            self.optimizer.step()
            
            # scheduler
            if self.scheduler:
                self.scheduler.step()
        else:
            pass