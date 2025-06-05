from BCNN.layers.misc import ModuleWrapper
from BCNN import layers
from torch import nn

#Implement a basic BBB architecture for MNIST


class BayesianCNN(ModuleWrapper):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        
        self.conv1 = layers.BBB_Conv2d(1, 6, 5, 1)
        self.relu1 = nn.ReLU()
        self.avg1 = nn.AvgPool2d(2, 2)
        self.conv2 = layers.BBB_Conv2d(6, 16, 5, 1)
        self.relu2 = nn.ReLU()
        self.avg2 = nn.AvgPool2d(2, 2)
        self.relu3 = nn.ReLU()
        self.flatten = layers.FlattenLayer(256)
        self.linear1 = layers.BBB_Linear(256, 120)
        self.relu4 = nn.ReLU()
        self.linear2 = layers.BBB_Linear(120, 84)
        self.relu5 = nn.ReLU()
        self.linaer3 = layers.BBB_Linear(84, 10)