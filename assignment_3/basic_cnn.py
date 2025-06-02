import torch
from torch import nn

#Implement a basic CNN architecture for MNIST

class basicCNN(nn.Module):
    def __init__(self):
        super().__init__()