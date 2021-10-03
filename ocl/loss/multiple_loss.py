from typing import List, Dict

import torch
from ocl.utils import create_instance


class MultipleLosses(torch.nn.Module):
    def __init__(self, losses: List[Dict], weights=None):
        super().__init__()
        losses = [create_instance(loss) for loss in losses]
        self.losses = torch.nn.ModuleList(losses)

        if weights is not None:
            self.assertions_if_not_none(weights)
            self.weights = weights
        else:
            self.weights = ([1] * len(losses))

    def forward(self, inputs, labels):
        total_loss = 0
        iterable = enumerate(self.losses)
        for i, loss_func in iterable:
            total_loss += (
                loss_func(inputs, labels) * self.weights[i]
            )
        return total_loss

    def assertions_if_not_none(self, x):
        if x is not None:
            assert len(x) == len(self.losses)