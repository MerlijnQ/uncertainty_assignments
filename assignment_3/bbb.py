import torch
from torch import nn

#Implement a basic BBB architecture for MNIST


class BBB(nn.Module):
    def __init__(self):
        super().__init__()