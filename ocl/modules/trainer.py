import torch.nn as nn
import torch

from ocl.utils import create_instance

class FinetuneHeadTrainer(object):
    def __init__(self,
                 criterion: dict=None,
                 optimizer: dict=None,
                 model: nn.Module=None,
                 lr_scheduler: dict=None,
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
        self.lr_scheduler = lr_scheduler
        
        if self.lr_scheduler:
            self.lr_scheduler = create_instance(lr_scheduler, optimizer=self.optimizer)
        
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
                
            x, y = x.to(device), y['label'].to(device)
            logits = self.model(x, **kwargs)
            loss += self.criterion(logits, y)
            
            # backward
            loss.backward()
            self.optimizer.step()
            
            # scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()
        else:
            pass