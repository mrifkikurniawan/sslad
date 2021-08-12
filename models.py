import torch
import torch.nn as nn

# multi-layer perceptron classifier
class MLP(nn.Module):
    def __init__(self, 
                 input_dims:int, 
                 num_classes:int):
        super(MLP, self).__init__()
        
        self.classifier = nn.Sequential(nn.BatchNorm1d(input_dims),
                                        nn.Dropout(p=0.25, inplace=False),
                                        nn.Linear(input_dims, 512, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm1d(512),
                                        nn.Dropout(p=0.5, inplace=False),
                                        nn.Linear(512, num_classes, bias=False))
        
        
        def forward(self, x):
            assert isinstance(x, torch.Tensor)
            
            out = self.classifier(x)
            return out