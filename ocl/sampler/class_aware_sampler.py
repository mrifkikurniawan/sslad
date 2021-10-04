import numpy as np
import random
from typing import List

import torch
from torch.utils.data import Sampler

class ClassIteratior(object):
    def __init__ (self, 
                  classes: list=None):
        super().__init__()
        
        self.classes = classes
        self.length = len(self.classes)
        self.i = self.length - 1
        
    def __iter__ (self):
        return self
    
    def __next__ (self) -> int:
        self.i += 1
        if self.i == self.length:
            self.i = 0
        return self.classes[self.i]
    
    
    
class DatasetIterator(object):
    def __init__ (self, data):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        if self.i == self.length:
            self.i = 0
            
        return self.data_list[self.i]
    
def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:        
        if j >= num_samples_cls:
            j = 0
        if j == 0:
            current_class = next(cls_iter)
            temp_tuple = next(zip(*[data_iter_list[current_class]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]  
        i += 1
        j += 1

class ClassAwareSampler(Sampler):
    def __init__(self, 
                 dataset: torch.utils.data.Dataset, 
                 sample_per_class: int=1,):
        
        num_classes = max(np.unique(dataset.get_labels())) + 1
        cls_data_list = [list() for _ in range(num_classes)]
        classes_for_iterator = list()
                
        for i, label in enumerate(dataset.get_labels()):
            cls_data_list[label].append(i)
        for i, label_datapoints in enumerate(cls_data_list):
            if label_datapoints != []:
                classes_for_iterator.append(i)
        self.class_iter = ClassIteratior(classes_for_iterator)
        self.data_iter_list = [DatasetIterator(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = sample_per_class
        
    def __iter__ (self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)
    
    def __len__ (self):
        return self.num_samples