import torch
import statistics
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from bbb import BayesianCNN
from metrics import metric

LEARNING_RATE = 1e-2
EPOCHS = 100
writer = SummaryWriter(log_dir="logs")

def Temp_scale():
    "Based on the calibration plots we can finetune our model. Optionally we can calculate ECE"
    pass

def set_seed():
    pass


def train_ensemble():
    pass


def train_bcnn(training_loader) -> BayesianCNN:
    # Returns a trained bcnn on the given training and validation loader
    model = BayesianCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = nn.CrossEntropyLoss()
    
    num_ens = 1
    for epoch in tqdm(range(EPOCHS), total=EPOCHS):
        train_results = []
        for x_batch, y_batch in training_loader:
            optimizer.zero_grad()
            outputs = torch.zeros(x_batch.shape[0], model.num_classes, num_ens)
            kl = 0.0
            
            for i in range(num_ens):
                model_output, _kl = model.forward(x_batch)
                kl += _kl
                outputs[:, :, i] = F.softmax(model_output, dim=1).data
                #Cross entropy already does softmax on the inside!!!!

            loss_value = loss(outputs, y_batch)
            loss_value.backward()
            
            train_results.append(loss_value.item())
            optimizer.step()
        median_loss = statistics.median(train_results)
        writer.add_scalar(f"Training loss - bcnn", median_loss, epoch)

    return model


@torch.no_grad
def test_bcnn(testing_loader, model: BayesianCNN):
    model.train(False)
    results = []

    num_ens = 10
    for x_batch, y_batch in testing_loader:
        outputs = torch.zeros(x_batch.shape[0], model.num_classes, num_ens)
        for i in range(num_ens):
            model_output, _kl = model.forward(x_batch)
            outputs[:, :, i] = F.softmax(model_output, dim=1).data
        
        results.append((outputs, y_batch))
    
    return results


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
    mnist_train_loader, mnist_test_loader, fmnist_train_loader, fmnist_test_loaderr = get_data()
    bayesian_cnn = train_bcnn(mnist_train_loader, mnist_test_loader)
    ensemble_cnn = train_ensemble(mnist_train_loader, mnist_test_loader)


if __name__ == "__main__":
    main()