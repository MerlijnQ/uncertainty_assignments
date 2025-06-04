import torch
import statistics
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from BCNN import metrics

from basic_cnn import basicCNN
from bbb import BayesianCNN
from metrics import metric

LEARNING_RATE = 1e-2
EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
writer = SummaryWriter(log_dir="logs")

def Temp_scale():
    "Based on the calibration plots we can finetune our model. Optionally we can calculate ECE"
    pass

def set_seed():
    pass


def train_ensemble(training_loader, n_models = 3):
    ensemble = []
    loss_function = nn.CrossEntropyLoss()
    
    for i in range(n_models):
        model = basicCNN()
        optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            model.train()
            losses = []
            correct = 0
            
            for x_batch, y_batch in training_loader:
                optimiser.zero_grad()
                
                outputs = model(x_batch)
                loss = loss_function(outputs, y_batch)
                loss.backward()
                optimiser.step()
                
                losses.append(loss.item())
                correct += (outputs.argmax(dim=1) == y_batch).sum().item
                
            accuracy = correct / len(training_loader.dataset)
            median_loss = statistics.median(losses)
            
            print(f"Epoch {epoch+1}/{EPOCHS}: loss = {median_loss:.4f} || accuracy = {accuracy:.4f}")
            
        ensemble.append(model)
        
    return ensemble


def test_ensemble(test_loader, ensemble: list[basicCNN]):
    for model in ensemble:
        model.eval()
        
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = torch.stack([
                F.softmax(model(x_batch), dim=1) for model in ensemble
            ])
            
            avg_output = outputs.mean(dim=0)
            
            preds = avg_output.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
    accuracy = correct / total
    
    return accuracy


def train_bcnn(training_loader) -> BayesianCNN:
    # Returns a trained bcnn on the given training and validation loader
    model = BayesianCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = nn.CrossEntropyLoss().to(DEVICE)
    training_loader = training_loader
    
    num_ens = 1
    for epoch in tqdm(range(EPOCHS), total=EPOCHS):
        train_results = []
        for x_batch, y_batch in training_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = torch.zeros(x_batch.shape[0], model.num_classes, num_ens, device=DEVICE)
            kl = 0.0
            
            for i in range(num_ens):
                model_output, _kl = model.forward(x_batch)
                kl += _kl
                outputs[:, :, i] = model_output
            outputs = outputs.mean(dim=2)

            # kl term is computed but not used?
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
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        outputs = torch.zeros(x_batch.shape[0], model.num_classes, num_ens)
        for i in range(num_ens):
            model_output, _kl = model.forward(x_batch)
            outputs[:, :, i] = model_output
        outputs = outputs.mean(dim=2)
        outputs = F.softmax(outputs, dim=1).data

        # Appends probability scores per y_batch, but don't understand the tensor itself
        # Actually I think first are all the probs and second is the highest
        results.append((outputs, y_batch))
        break
    
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
    mnist_train_loader, mnist_test_loader, fmnist_train_loader, fmnist_test_loader = get_data()
    bayesian_cnn = train_bcnn(mnist_train_loader)
    results_bayesian = test_bcnn(fmnist_test_loader, bayesian_cnn)
    print(results_bayesian)
    
    # ensemble_cnn = train_ensemble(mnist_train_loader, mnist_test_loader)
    mnist_train_loader, mnist_test_loader, fmnist_train_loader, fmnist_test_loaderr = get_data()
    # bayesian_cnn = train_bcnn(mnist_train_loader, mnist_test_loader)
    ensemble_cnn = train_ensemble(mnist_train_loader, mnist_test_loader)


if __name__ == "__main__":
    main()