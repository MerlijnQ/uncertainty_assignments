import torch
import statistics
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from BCNN import metrics
from BCNN import utils

from basicCNN import basicCNN
from bbb import BayesianCNN

LEARNING_RATE = 1e-2
EPOCHS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
writer = SummaryWriter(log_dir="logs")


class TrainInference():
    def __init__(self):
        print("Ready to get some numbers")

    def train_ensemble(self, training_loader, val_loader, n_models = 3):
        ensemble = []
        loss_function = nn.CrossEntropyLoss()
        
        for i in range(n_models):
            model = basicCNN()
            optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            for epoch in tqdm(range(EPOCHS)):
                model.train()
                
                losses = []
                for x_batch, y_batch in training_loader:
                    optimiser.zero_grad()
                    
                    outputs = model(x_batch)
                    loss = loss_function(outputs, y_batch)
                    loss.backward()
                    optimiser.step()
                    
                    losses.append(loss.item())
                
                mean_loss = statistics.mean(losses)
                writer.add_scalar(f"Training loss - Ensemble CNN {i}", mean_loss, epoch)
                
                losses = []
                for x_batch, y_batch in val_loader:
                    optimiser.zero_grad()
                    
                    outputs = model(x_batch)
                    loss = loss_function(outputs, y_batch)
                    loss.backward()
                    optimiser.step()
                    
                    losses.append(loss.item())
                
                mean_loss = statistics.mean(losses)
                writer.add_scalar(f"Validation loss - Ensemble CNN {i}", mean_loss, epoch)
                
            ensemble.append(model)
            
        return ensemble


    def train_bcnn(self, training_loader, val_loader) -> BayesianCNN:
        # Returns a trained bcnn on the given training loader
        model = BayesianCNN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_train = metrics.ELBO(len(training_loader)).to(DEVICE)
        loss_val = metrics.ELBO(len(val_loader)).to(DEVICE)
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

                outputs = utils.logmeanexp(outputs, dim=2).to(DEVICE)
                kl = kl / num_ens

                beta = metrics.get_beta(i-1, len(training_loader), "standard", epoch, EPOCHS)
                loss_value = loss_train(outputs, y_batch, kl, beta)
                loss_value.backward()
                optimizer.step()
                
                train_results.append(loss_value.item())
            mean_loss = statistics.mean(train_results)
            writer.add_scalar("Training loss - bcnn", mean_loss, epoch)

            # Val loop
            val_results = []
            for i, (x_batch, y_batch) in enumerate(val_loader):
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = torch.zeros(x_batch.shape[0], model.num_classes, num_ens)

                kl = 0.0
                for i in range(num_ens):
                    model_outputs, _kl = model.forward(x_batch)
                    kl += _kl
                    outputs[:, :, i] = F.log_softmax(model_outputs, dim=1).data

                kl = kl / num_ens
                outputs = utils.logmeanexp(outputs, dim=2).to(DEVICE)

                beta = metrics.get_beta(i-1, len(val_loader), "standard", epoch, EPOCHS)
                loss_value = loss_val(outputs, y_batch, kl, beta)
                val_results.append(loss_value.item())
            mean_loss = statistics.mean(val_results)
            writer.add_scalar("Validation loss - bcnn", mean_loss, epoch)

        return model


    @torch.no_grad
    def bayesian_inference(self, test_loader, model: BayesianCNN):
        model.eval()
        model.to(DEVICE)
        mean_probs = []
        labels = []

        for x_batch, y_batch in test_loader:
            logits = []
            x_batch = x_batch.to(DEVICE)
            labels.append(y_batch.detach().cpu().numpy())
            output = model(x_batch)

            if isinstance(output, tuple):
                output, _ = output

            logits.append(output.detach().cpu())
            probs = F.softmax(torch.stack(logits), dim=-1)
            mean_probs.append(torch.mean(probs, dim=0))

        all_probs = torch.cat(mean_probs, dim=0)  # [total_samples, n_classes]
        labels = np.concatenate(labels)

        return all_probs.numpy(), labels


    @torch.no_grad
    def ensemble_inference(self, test_loader, models):
        for model in models:
            model.to(DEVICE)
            model.eval()

        mean_probs = []
        labels = []

        for data, label in test_loader:
            logits = []
            data = data.to(DEVICE)
            labels.append(label.detach().cpu().numpy())
            for model in models:
                logits.append(model(data).detach().cpu())

            probs = F.softmax(torch.stack(logits), dim=-1)
            mean_probs.append(torch.mean(probs, dim=0))
        all_probs = torch.cat(mean_probs, dim=0)  # [total_samples, n_classes]
        labels = np.concatenate(labels)

        return all_probs.numpy(), labels