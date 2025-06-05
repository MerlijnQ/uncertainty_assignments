import torch
from torch import nn
import torch.nn.functional as F

#Implement a basic CNN architecture for MNIST

class basicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.relu1 = nn.ReLU()
        self.avg1 = nn.AvgPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.relu2 = nn.ReLU()
        self.avg2 = nn.AvgPool2d(2, 2)
        
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        
        self.linear1 = nn.Linear(256, 120)
        self.relu4 = nn.ReLU()
        
        self.linear2 = nn.Linear(120, 84)
        self.relu5 = nn.ReLU()
        
        self.linear3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.avg1(x)
        
        x = self.relu2(self.conv2(x))
        x = self.avg2(x)
        
        x = self.relu3(x)
        x = self.flatten(x)
        
        x = self.relu4(self.linear1(x))
        x = self.relu5(self.linear2(x))
        x = self.linear3(x)
        
        return x