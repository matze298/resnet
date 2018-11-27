# ResNet
ResNet implementation using tensorflow.

Currently, the ResNet-110 for CIFAR-10 is implemented.
The first goal is to match the reported results of the paper on CIFAR-10 (see table below). 

## Results

Dataset | Results | Reported Results
------- | ------------------ | -----------------
CIFAR-10 | 81.25%             | 93.67 %


## Differences to paper
* Using ADAM-optimizer instead of Momentum-SGD
* Slightly different learning rate schedule
* Smaller batch size due to GPU-limitations
