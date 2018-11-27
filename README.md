# ResNet
My own ResNet implementation using tensorflow. 

Currently, the ResNet-110 for CIFAR-10 is implemented.
The first goal is to match the reported results of the paper (see table below). 

*Differences to paper:*
* Using ADAM-optimizer instead of Momentum-SGD

Results:
Dataset | Results (Test Set) | Reported Results
------- | ------------------ | -----------------
CIFAR-10 | 81.25%             | 93.67 %
