from typing import Dict, List
from copy import deepcopy
from tqdm import tqdm
from PIL import Image

from randaugment import RandAugment
import pandas as pd
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from ocl.utils import *



class RMSampler(object):
    def __init__(self, 
                 augmentation: str="vr_randaug",
                 batch_size: int=32,
                 num_workers: int=32):
        
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.handlers = list()
    
    def __call__(self, dataset: Dataset, num_samples: int, model: nn.Module, target_layer: str) -> Image:
        
        if num_samples > len(dataset):
            num_samples = len(dataset)
        
        # register forward hook to get features
        self.target_layer = target_layer
        self._register_forward_hook(model)
        
        uncertainy_score_per_sample, outputs = self._montecarlo(dataset, model)
        sample_df = pd.DataFrame(uncertainy_score_per_sample)
        
        selected_samples_indices = self._select_indices(sample_df=sample_df, num_samples=num_samples)
        assert len(selected_samples_indices) == num_samples
        
        # storage the tensor samples to list
        selected_images = [dataset.inputs[idx] for idx in selected_samples_indices]
        selected_targets = [dataset.targets[idx] for idx in selected_samples_indices]
        logits = outputs['logits']
        features = outputs['features']
        for i, idx in enumerate(selected_samples_indices):
            selected_targets[i]['logit'] = logits[idx]
            selected_targets[i]['feature'] = features[idx]
        
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

    def _select_indices(self, sample_df: pd.DataFrame, num_samples: int) -> List[int]:
        uncertain_samples = sample_df.sort_values(by="uncertainty", ascending=False)
        selected_samples_indices = uncertain_samples.index[0:num_samples].tolist()

        return selected_samples_indices
    
    def _init_augmentation(self) -> List[callable]:
        transform_cands = list()
        if self.augmentation == "vr_randaug":
            for _ in range(12):
                transform_cands.append(RandAugment())
        
        return transform_cands         

    def _montecarlo(self, dataset: Dataset, model: nn.Module) -> List[Dict]:
        # initialize list of augmenters
        augmentation_module = self._init_augmentation()
        n_transforms = len(augmentation_module)
        uncertainty_scores_per_augment = dict()
        uncertainy_score_per_sample = list()
        
        # inference on each single augmenter to obtain uncertainty score per augmenter
        for idx, tr in enumerate(augmentation_module):
            _tr = transforms.Compose([tr] + [transforms.ToTensor(), transforms.Normalize((0.3252, 0.3283, 0.3407), (0.0265, 0.0241, 0.0252))])
            uncert_name = f"uncert_{str(idx)}"
            uncertainty_scores_per_augment[uncert_name], labels, outputs = self._compute_uncert(dataset, _tr, model=model)
        
        len_samples = len(dataset)
        for i in range(len_samples):
            score = dict(label=labels[i])
            for uncert_name in uncertainty_scores_per_augment:
                unc_scores_sample_i = uncertainty_scores_per_augment[uncert_name][i]
                score[uncert_name] = unc_scores_sample_i
            uncertainy_score_per_sample.append(score)
        
        for i in range(len_samples):
            sample = uncertainy_score_per_sample[i]
            self._variance_ratio(sample, n_transforms)  
            
        return uncertainy_score_per_sample, outputs
        
    def _compute_uncert(self, dataset: Dataset, transforms: List[callable], model: nn.Module) -> List[torch.Tensor]:
        batch_size = self.batch_size
        original_dataset_transform = deepcopy(dataset.transform)
        dataset.transform = transforms

        uncertainty_scores = list()
        labels = list()
        logits = list()
        features = list()
        outputs = dict()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        
        model.eval()
        with torch.no_grad():
            for _, mbatch in tqdm(enumerate(dataloader), desc=f"measure uncertainty"):
                try:
                    x, y, _ = mbatch
                except:
                    x, y = mbatch
                x = x.type_as(next(model.parameters()))
                logit = model(x)
                logit = logit.detach().cpu()

                for i, cert_value in enumerate(logit):
                    uncertainty_value = 1 - cert_value
                    uncertainty_scores.append(uncertainty_value)
                    labels.append(y['label'][i].item())
                    logits.append(cert_value)
                    features.append(self.features[self.target_layer][i].detach().cpu().squeeze())

        # return back the original transform
        dataset.transform = original_dataset_transform
        outputs['logits'] = logits
        outputs['features'] = features    
        return uncertainty_scores, labels, outputs

    def _variance_ratio(self, sample, cand_length):
        vote_counter = torch.zeros(sample["uncert_0"].size(0))
        for i in range(cand_length):
            top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item()
        
    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()