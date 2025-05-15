import torch
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

Mean = 0.0
SD_1 = 0.19
SD_2 = 0.33
device = "cuda" if torch.cuda.is_available() else "cpu"


class drop_connect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, p=0.2):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.p = p
        self.inference = False

    def forward(self, x):
        if self.inference:
            weight = F.dropout(self.weight, p=self.p, training=True)
        else:
            weight = self.weight
        return F.linear(x, weight, self.bias)


class regression_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.regression_layer = nn.Sequential(
            drop_connect(1, 32, p=0.2),
            nn.ReLU(),
            drop_connect(32, 32, p=0.2),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(32, 1)
        self.variance_head = nn.Linear(32, 1)

    def forward(self, x):
        x = self.regression_layer(x)
        mean = self.mean_head(x)
        variance = F.softplus(self.variance_head(x))
        return mean, variance


def train_model(model, x, y, fase_1=True, epochs=100, learning_rate=0.01):
    print(f"Training with {"MSE" if fase_1 else "Gaussian NLL"}")
    writer = SummaryWriter(log_dir="logs")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    criterion_2 = nn.GaussianNLLLoss()

    if fase_1:
        for param in model.variance_head.parameters():
            param.requires_grad = False
    else:
        for param in model.variance_head.parameters():
            param.requires_grad = True

    x_tensors = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device=device)
    y_tensors = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device=device)
    model = model.to(device=device)

    batch_size = 32
    dataset = torch.utils.data.TensorDataset(x_tensors, y_tensors)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in tqdm(range(epochs), total=epochs):
        results = []
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            model_mean, model_variance = model(x_batch)

            if fase_1:
                loss = criterion(model_mean, y_batch)
            else:
                loss = criterion_2(model_mean, y_batch, model_variance)

            results.append(loss.item())
            loss.backward()
            optimizer.step()
        median_loss = statistics.median(results)
        writer.add_scalar(f"Loss/train_{"MSE" if fase_1 else "GNLL Loss"}", median_loss, epoch)
        # print(f"Epoch {epoch}, Loss: {median_loss}")

    return model


def Gaussian(x, mean, SD):
    dist = (1/(SD*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mean) / SD)**2)
    return dist


def sample_distribution(x):
    dist =  x * np.sin(x) + Gaussian(x, Mean, SD_1) + Gaussian(x, Mean, SD_2) * x
    return dist


def disentangle(mean, variance):
    epistemic = mean.var(dim=0)
    aleatoric = variance.mean(dim=0)
    total_uncertainty = epistemic + aleatoric
    return epistemic, aleatoric, total_uncertainty


def enable_dropconnect(model):
    for m in model.modules():
        if isinstance(m, drop_connect):
            m.inference = True


def dropout_inference(model, x, T=50):
    model.eval()
    enable_dropconnect(model)
    x: torch.Tensor
    x = x.to(device=device)
    
    epistemic = []
    aleatoric = []
    total_uncertainty = []
    
    for x_i in tqdm(range(x.shape[0]), total=x.shape[0]):
        x_cur = x[x_i]
        mean_predictions = []
        variance_predictions = []
        for _ in range(T):
            mean_pred, var_pred = model(x_cur)
            mean_predictions.append(mean_pred.cpu().item())
            variance_predictions.append(var_pred.cpu().item())
        
        # Disentangle uncertainty
        epistemic_uncertainty = statistics.variance(mean_predictions)
        aleatoric_uncertainty = statistics.mean(variance_predictions)
        epistemic.append(epistemic_uncertainty)
        aleatoric.append(aleatoric_uncertainty)
        total_uncertainty.append(epistemic_uncertainty + aleatoric_uncertainty)
    
    df = pd.DataFrame({
        "x_values": x.squeeze().cpu().numpy(),
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "total_uncertainty": total_uncertainty,
    })
    
    return df


def main():
    # Does the model learn the variance or the mean, cause otherwise we need to take the exp of the variance output?
    #  I assume the model is supposed to learn the distribution of the data?

    # Experiment setup
    np.random.seed(42)
    torch.manual_seed(42)
    x = np.linspace(-10, 10, 1000)
    y = sample_distribution(x)
    OOD = np.linspace(-20, 20, 1000)

    # Create and train model
    model = regression_model()
    model = train_model(model, x, y, fase_1=True, epochs=100)
    model = train_model(model, x, y, fase_1=False, epochs=100)

    df = dropout_inference(model, torch.tensor(OOD, dtype=torch.float32).view(-1, 1))
    print(df)

    """Make plots for both predictive
    and each source of uncertainty, and argue/describe/analyze/discuss if each uncertainty estimate makes
    sense or not. For this use your in-distribution and out of distribution datasets"""
    #To see if it performs well we can concatonate the OOD and ID data, create grond truth labels and plot the AUROC which should be high -> SkLearn has a build in AUROC function for this
    #Furthermore we can plot the graphs for both uncertainties for OOD and ID and see the differfence

    # Plot all the different uncertainties
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(df['x_values'], df['epistemic'], label='Epistemic', color='blue')
    axes[0].set_title('Epistemic Uncertainty')
    axes[0].set_xlabel('x_values')
    axes[0].set_ylabel('Uncertainty')
    axes[0].grid(True)
    axes[1].plot(df['x_values'], df['aleatoric'], label='Aleatoric', color='green')
    axes[1].set_title('Aleatoric Uncertainty')
    axes[1].set_xlabel('x_values')
    axes[1].set_ylabel('Uncertainty')
    axes[1].grid(True)
    axes[2].plot(df['x_values'], df['total_uncertainty'], label='Total Uncertainty', color='red')
    axes[2].set_title('Total Uncertainty')
    axes[2].set_xlabel('x_values')
    axes[2].set_ylabel('Uncertainty')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("uncertainties_plots.png")


if __name__ == '__main__':
    main()
