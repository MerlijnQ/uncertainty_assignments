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
from BCNN import utils

from basic_cnn import basicCNN
from bbb import BayesianCNN
from metrics import metric

LEARNING_RATE = 1e-2
BATCH_SIZE = 64
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
            
            for x_batch, y_batch in tqdm(training_loader):
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


@torch.no_grad
def test_ensemble(test_loader, ensemble: list[basicCNN]):
    for model in ensemble:
        model.eval()
        
    correct = 0
    total = 0
    pred_target_pairs = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = torch.stack([
                F.softmax(model(x_batch), dim=1) for model in ensemble
            ])
            
            avg_output = outputs.mean(dim=0)
            preds = avg_output.argmax(dim=1)
            
            pred_target_pairs.extend(zip(preds.tolist(), y_batch.tolist()))
            
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
    accuracy = correct / total
    
    return pred_target_pairs, accuracy


def train_bcnn(training_loader, val_loader) -> BayesianCNN:
    # Returns a trained bcnn on the given training loader
    model = BayesianCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = metrics.ELBO(BATCH_SIZE).to(DEVICE)
    training_loader = training_loader
    
    num_ens = 1
    for epoch in tqdm(range(EPOCHS), total=EPOCHS):
        # Train loop
        train_results = []
        for i, (x_batch, y_batch) in enumerate(training_loader):
            optimizer.zero_grad()

            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = torch.zeros(x_batch.shape[0], model.num_classes, num_ens, device=DEVICE)
            
            kl = 0.0
            for i in range(num_ens):
                model_output, _kl = model.forward(x_batch)
                kl += _kl
                outputs[:, :, i] = F.log_softmax(model_output, dim=1)

            outputs = utils.logmeanexp(outputs, dim=2)
            kl = kl / num_ens

            beta = metrics.get_beta(i-1, BATCH_SIZE, "standard", epoch, EPOCHS)
            loss_value = loss(outputs, y_batch, kl, beta)
            loss_value.backward()
            optimizer.step()
            
            train_results.append(loss_value.item())
        mean_loss = statistics.mean(train_results)
        writer.add_scalar(f"Training loss - bcnn", mean_loss, epoch)

        # Val loop (TODO)
        val_results = []
        for i, (x_batch, y_batch) in enumerate(val_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = torch.zeros(x_batch.shape[0], model.num_classes, num_ens)

            for i in range(num_ens):
                model_outputs, _ = model.forward(x_batch)
                outputs[:, :, i] = F.log_softmax(model_outputs, dim=1).data

            outputs = utils.logmeanexp(outputs, dim=2)

            beta = metrics.get_beta(i-1, BATCH_SIZE, "standard", epoch, EPOCHS)
            loss_value = loss(outputs, y_batch, kl, beta)
            val_results.append(loss_value.item())
        mean_loss = statistics.mean(val_results)
        writer.add_scalar(f"Validation loss - bcnn", mean_loss, epoch)

    return model


@torch.no_grad
def test_bcnn(testing_loader, model: BayesianCNN):
    model.train(False)
    results = []
    accs = []

    num_ens = 10
    for i, (x_batch, y_batch) in enumerate(testing_loader):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        outputs = torch.zeros(x_batch.shape[0], model.num_classes, num_ens)

        for i in range(num_ens):
            model_output, _ = model.forward(x_batch)
            outputs[:, :, i] = model_output
        
        outputs = utils.logmeanexp(outputs, dim=2)
        print(outputs.shape)
        outputs = F.softmax(outputs, dim=1)
        print(outputs)
        accs.append(metrics.acc(outputs, y_batch))
        # results.append((outputs, y_batch))
        break
    
    return accs


def inference():
    pass


def deep_ensemble():
    pass


def get_data(val_ratio=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and normalize to [0, 1]
    ])

    # Load full training datasets
    mnist_full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_full_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    # Calculate split sizes
    mnist_val_size = int(len(mnist_full_train) * val_ratio)
    mnist_train_size = len(mnist_full_train) - mnist_val_size

    fmnist_val_size = int(len(fmnist_full_train) * val_ratio)
    fmnist_train_size = len(fmnist_full_train) - fmnist_val_size

    # Split the datasets
    mnist_train, mnist_val = random_split(mnist_full_train, [mnist_train_size, mnist_val_size])
    fmnist_train, fmnist_val = random_split(fmnist_full_train, [fmnist_train_size, fmnist_val_size])

    # Load test datasets
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    mnist_train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
    mnist_val_loader = DataLoader(mnist_val, batch_size=BATCH_SIZE, shuffle=False)
    mnist_test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

    fmnist_train_loader = DataLoader(fmnist_train, batch_size=BATCH_SIZE, shuffle=True)
    fmnist_val_loader = DataLoader(fmnist_val, batch_size=BATCH_SIZE, shuffle=False)
    fmnist_test_loader = DataLoader(fmnist_test, batch_size=BATCH_SIZE, shuffle=False)

    return (
        mnist_train_loader, mnist_val_loader, mnist_test_loader,
        fmnist_train_loader, fmnist_val_loader, fmnist_test_loader
    )

def main():
    (mnist_train_loader, mnist_val_loader, mnist_test_loader,
     fmnist_train_loader, fmnist_val_loader, fmnist_test_loader) = get_data()
    bayesian_cnn = train_bcnn(mnist_train_loader, mnist_val_loader)
    results_bayesian = test_bcnn(mnist_test_loader, bayesian_cnn)
    print(results_bayesian)
    exit()
    
    ensemble_cnn = train_ensemble(mnist_train_loader, mnist_test_loader)


if __name__ == "__main__":
    main()