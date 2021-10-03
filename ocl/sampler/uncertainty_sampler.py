from typing import Dict, List

import torch
import torch.nn as nn
from ocl.utils import *
import scoring


class UncertaintySampler(object):
    def __init__(self, 
                 num_workers: int=32,
                 scoring_method: str="entropy",
                 negative_mining: bool=True):
        
        self.num_workers = num_workers
        self.scoring = getattr(scoring, scoring_method)
        self.negative_mining = negative_mining
        self.handlers = list()
    
    def __call__(self, x: torch.Tensor, y: Dict[str, torch.Tensor], model: nn.Module, num_samples: int, target_layer: str) -> List[torch.Tensor]:
        len_inputs = x.shape[0]
        if num_samples > len_inputs:
            num_samples = len_inputs

        # register forward hook to get features
        self.target_layer = target_layer
        self._register_forward_hook(model)
 
        labels = y['label']
        samples_scores, model_outputs = self._compute_score(x, labels, model)        
        selected_samples_indices = self._select_indices(samples_scores, num_samples=num_samples)
        
        assert len(selected_samples_indices) == num_samples
        selected_images = x[selected_samples_indices]
        
        # convert batch tensor to list of tensor
        selected_images = [x.cpu() for x in selected_images]
        selected_targets = [dict(label=y['label'][i].cpu(), 
                                 logit=model_outputs['logits'][i].cpu(), 
                                 feature=model_outputs['features'][i].cpu()) for i in selected_samples_indices]
        
        # remove hook
        self.remove_hook()
        
        return selected_images, selected_targets

    def _register_forward_hook(self, model: nn.Module):
        self.features = dict()
        def get_features(key):
            def forward_hook(module, input, output):
                self.features[key] = output#.detach()
            return forward_hook
        
        # add forward hook 
        for name, module in model.named_modules():
            if self.target_layer == name:
                self.handlers.append(module.register_forward_hook(get_features(key=name)))

    def _select_indices(self, samples_scores: torch.Tensor, num_samples: int) -> torch.Tensor:
        selected_samples_indices = torch.topk(samples_scores, k=num_samples, dim=0, largest=False)[1]

        return selected_samples_indices        

    def _compute_score(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> List[Dict]:
        
        # inference to get uncertainty scores
        # given latest updated model
        model.eval()
        logits = list()
        features = list()
        outputs = dict()
        
        with torch.no_grad():
            logit = model(x)
            proba_dist = nn.functional.softmax(logit, dim=1)

            # computing sampling scores based on uncertainty 
            # and negative mining if necessary
            samples_scores = self.scoring(proba_dist)
            if self.negative_mining:
                negative_scores = scoring.negative_scoring(proba_dist, y)
                samples_scores += negative_scores
            
            for i in range(x.shape[0]):
                logits.append(logit[i].detach())
                features.append(self.features[self.target_layer][i].detach().squeeze())
        
        outputs['logits'] = logits
        outputs['features'] = features
        return samples_scores, outputs        
                
    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove() 