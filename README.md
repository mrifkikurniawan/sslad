### ICCV 2021 Workshop SSLAD Track 3A - Continual Learning Classification

See the Codalab challenge for the challenge details
[here](https://competitions.codalab.org/competitions/33830).

### File overview

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
