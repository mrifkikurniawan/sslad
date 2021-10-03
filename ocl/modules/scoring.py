import torch
from torch import nn

def margin_conf(proba_dist: torch.Tensor):  
    """[measure the uncertainty of samples given its probability distribution based on 
        margin between most confidence and second most confidence scores]

    Args:
        proba_dist (torch.Tensor): [probability distribution. dim: (num_samples, n_class)]

    Returns:
        uncertainty_score [torch.Tensor]: [uncertainty scores. dim: (num_samples,)]
    """      
    values, _ = torch.topk(proba_dist, k=2, dim=1)
    most_conf = values[:, 0]
    next_most_conf = values[:, 1]
    uncertainty_score = 1 - (most_conf - next_most_conf)
    
    return uncertainty_score

def least_conf(proba_dist: torch.Tensor):
    """[measure uncertainty value based on most confidence probabilty confidence score]

    Args:
        proba_dist (torch.Tensor): [probability distribution. dim: (num_samples, n_class)]

    Returns:
        uncertainty_score [torch.Tensor]: [uncertainty scores. dim: (num_samples,)]
    """    
    n_class = len(proba_dist)
    most_conf, _ = torch.max(proba_dist, dim=1)
    uncertainty_score = (1 - most_conf) * n_class / (n_class - 1)
    
    return uncertainty_score

def entropy(proba_dist: torch.Tensor):
    """[measure the uncertainty given sample probability distribution]

    Args:
        proba_dist (torch.Tensor): [probability distribution. dim: (num_samples, n_class)]

    Returns:
        uncertainty_score [torch.Tensor]: [uncertainty scores. dim: (num_samples,)]
    """    
    log_probs = torch.log(proba_dist)
    uncertainty_score = (proba_dist * -log_probs).sum(dim=1)
    
    return uncertainty_score

def negative_scoring(proba_dist: torch.Tensor, y: torch.Tensor):
    score = 0.5
    _, preds = torch.max(proba_dist, dim=1)
    incorrect_preds = preds != y
    
    return incorrect_preds.type(torch.float16) * score