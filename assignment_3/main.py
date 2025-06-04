import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from bbb import BayesianCNN
from metrics import metric

def Temp_scale():
    "Based on the calibration plots we can finetune our model. Optionally we can calculate ECE"
    pass

def setSeed():
    pass


def train():
    pass


def inference():
    pass


def deepEnsemble():
    pass


def getData():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and normalize to [0, 1]
    ])

    # Load MNIST dataset
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Load Fashion MNIST dataset
    fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    mnist_train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

    fmnist_train_loader = DataLoader(fmnist_train, batch_size=64, shuffle=True)
    fmnist_test_loader = DataLoader(fmnist_test, batch_size=64, shuffle=False)
    
    return mnist_train_loader, mnist_test_loader, fmnist_train_loader, fmnist_test_loader


def main():
    pass


if __name__ == "__main__":
    main()