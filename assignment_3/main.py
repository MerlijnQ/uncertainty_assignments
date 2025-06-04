import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from bbb import BayesianCNN


def set_seed():
    pass


def train(training_loader, testing_loader, model):
    pass


def inference():
    pass


def deep_ensemble():
    pass


def get_data():
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
    # Setup
    mnist_train_loader, mnist_test_loader, fmnist_train_loader, fmnist_test_loader = get_data()
    bayesian_cnn = BayesianCNN()


if __name__ == "__main__":
    main()