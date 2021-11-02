<div align="center">    
 
# Online Continual Learning via Multiple Deep Metric Learning and Uncertainty-guided Episodic Memory Replay     
## 3<sup>rd</sup> Place Solution for ICCV 2021 Workshop SSLAD Track 3A - Continual Learning Classification

[![Technical Report](https://img.shields.io/badge/technical%20report-pdf-red)](https://intip.in/xjtusslad3a/)
[![slides](https://img.shields.io/badge/slides-ppt-orange)](https://intip.in/pptxjtusslad3a)  
[![video](https://img.shields.io/badge/video-youtube-critical)](https://youtu.be/MANuneF0DMw?t=17017)  
</div>
 
## Description   
Official implementation of our solution (3<sup>rd</sup> place) for ICCV 2021 Workshop Self-supervised Learning for Next-Generation Industry-level Autonomous Driving (SSLAD) Track 3A - Continual Learning Classification using *"Online Continual Learning via Multiple Deep Metric Learning and Uncertainty-guided Episodic Memory Replay"*. 

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/mrifkikurniawan/sslad.git

# install project   
cd sslad 
pip3 install -r requirements.txt   
 ```   

Next, preparing the dataset via links below.
- Train/val: [SODA10M Labeled Trainval (Google Drive)](https://drive.google.com/file/d/1oSJ0rbqNHLmlOOzpmQqXLDraCCQss4Q4/view). Then, replace the original annotations with this new annotations with timestamps: [[train](https://drive.google.com/file/d/1S5x45uDX6O1KvbYH30X_yW9G-uGX7QcT/view?usp=sharing)][[val](https://drive.google.com/file/d/1PLGugKgp4j7sruiRJxqGb-WzTyavh2kU/view?usp=sharing)].
- Test: [SODA10M 3A Tesetset (Google Drive)](https://drive.google.com/file/d/14IRVJlcIBUHt3v79kyqCodU-ZQMOMOCp/view?usp=sharing)

 Next, run training.   
 ```bash

# run training module with our proposed cl strategy
python3.9 classification.py \
--config configs/cl_strategy.yaml \
--name {path/to/log} \
--root {root/of/your/dataset} \
--num_workers {num workers} \
--gpu_id {your-gpu-id} \
--comment {any-comments} 
--store \
```
or see the [train.sh](train.sh) for the example.

## Experiments Results
| Method | Val AMCA | Test AMCA |
| ----------- | ----------- | ----------- | 
| Baseline (Uncertainty Replay)<sup>*</sup> | 57.517 | - |
| + Multi-step Lr Scheduler<sup>*</sup> | 59.591 (+2.07) | - |
| + Soft Labels Retrospection<sup>*</sup> | 59.825 (+0.23) | - |
| + Contrastive Learning<sup>*</sup> | 60.363 (+0.53) | 59.68 |
| + Supervised Contrastive Learning<sup>*</sup> | 61.49 (+1.13) | - |
| + Change backbone to ResNet50-D<sup>*</sup> | 62.514 (+1.02) | - |
| + Focal loss<sup>*</sup> | 62.71 (+0.19) | - |
| + Cost Sensitive Cross Entropy | 63.33 (+0.62) | - |
| + Class Balanced Focal loss<sup>*</sup> | **64.01 (+1.03)** | **64.53 (+4.85)** |
| + Head Fine-tuning with Class Balanced Replay | <span style="color:red">65.291 (+1.28)</span> | <span style="color:red">62.58 (-1.56)</span> |
| + Head Fine-tuning with Soft Labels Retrospection | <span style="color:red">66.116 (+0.83)</span> | <span style="color:red">62.97 (+0.39)</span> |
<sup>*</sup>Applied to our final method.

## File overview

`classification.py`: Driver code for the classification subtrack. 
There are a few things that can be changed here, such as the
model, optimizer and loss criterion. There are several arguments that can be set to store 
results etc. (Run `classification.py --help` to get an overview, or check the file.)

`class_strategy.py`: Provides an empty plugin. Here, you can define
your own strategy, by implementing the necessary callbacks. Helper
methods and classes can be ofcourse implemented as pleased. See
[here](https://github.com/VerwimpEli/avalanche/tree/master/avalanche/training/plugins)
for examples of strategy plugins.

`data_intro.ipynb`: In this notebook the stream of data is further introduced and explained.
Feel free to experiment with the dataset to get a good feeling of the challenge.

*Note: not all callbacks
have to be implemented, you can just delete those that you don't need.* 

`classification_util.py` & `haitain_classification.py`: These files contain helper code for 
dataloading etc. There should be no reason to change these.
