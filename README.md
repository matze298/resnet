# ResNet
Modular ResNet implementation using tensorflow.
PreActivation approach is used for all residual or bottleneck-blocks. 

Currently, the ResNet-110 for CIFAR-10 is implemented.
The first goal is to match the reported results of the paper on CIFAR-10 (see table below). 

## Results
The training set is split into 45k training images and 5k validation images. Results are reported on the test set. 

Dataset  | Layers | Results | Optimizer | Reported Results
-------- | ------ |-------- | --------- | -----------------
CIFAR-10 | 110    | 83.28%  | Adam      | 93.67 %
CIFAR-10 | 110    | 91.36%  | Momentum-SGD | 93.67 %


## Differences to paper
* batch size 32 due to GPU-limitations
* Learning rate schedule adapted due to different batch size
