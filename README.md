# Image Style-Transfer-Pytorch
This Repo is the implementation of the following paper:

* [AdaIN](https://arxiv.org/abs/1703.06868) Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

requirements.txt contains the information about all the libraries needed. Open terminal in the current path and do the following:
```
1. create virtual environment using:
    mkvirtualenv virtual-py2 -p python2
2. pip install -r requirements.txt
```
I used 2 Titan Xp GPUs. 

### Dataset 

```
./data-local/bin/prepare_cifar10.sh
```

###  Accuracy Achieved on Test Dataset

![alt text](https://github.com/iSarmad/Style-Transfer-Pytorch/blob/master/Result%20Images/wstyle%20one/Test/alpha1.png)



## Running the Training 





## Visdom Visualization
To Visualize on Visdom, use the following command 
```
visdom
```
Now you can start training or testing by running train.py or test.py. Visualize the output by going to the following address on your browser:

```
http://localhost:8097/
```


## License

This project is licensed under the MIT License. 
For specific helper function used in this repository please see the license agreement of the Repo linked in Acknowledgement section

## Acknowledgments
My implementation has been inspired from the following sources.

* [naoto0804](https://github.com/naoto0804/pytorch-AdaIN) : An unofficial implementation in Pytorch
* [xunhuang1995](https://github.com/xunhuang1995/AdaIN-style) - Official Implementation in Torch
