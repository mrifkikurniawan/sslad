from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxT(nn.Module):

    def __init__(self, temperature: Union[float, int, torch.Tensor]) -> None:
        super(SoftmaxT, self).__init__()

        if isinstance(temperature, float) or isinstance(temperature, int):
            temperature = torch.tensor([temperature])
        self.temperature = temperature

    def forward(self, inputs: torch.Tensor, act_f: str) -> torch.Tensor:
        temperature = self.temperature.type_as(inputs)
        out = inputs/temperature

        if act_f == 'softmax':
            out = F.softmax(out, dim=1)
        elif act_f == 'log_softmax':
            out = F.log_softmax(out, dim=1)
        return out  