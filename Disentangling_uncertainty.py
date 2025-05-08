import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

Mean = 0.0
SD_1 = 0.19
SD_2 = 0.

class drop_connect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, p=0.2):
        super(drop_connect, self).__init__(in_features, out_features, bias)
        self.p = p
        self.force_drop = True

    def forward(self, x):
        if self.training or self.force_drop:
            weight = F.dropout(self.weight, p=self.p, training=True)
        else:
            weight = self.weight
        return F.linear(x, weight, self.bias)

class regression_model(nn.Module):
    def __init__(self):
        super(regression_model, self).__init__()
        self.regression_layer = nn.Sequential(
            F.relu(drop_connect(1, 32, p=0.2)),
            F.relu(drop_connect(32, 32, p=0.2)),
        )

        self.mean_head = nn.Linear(32, 1)
        self.variance_head = nn.Linear(32, 1)


    def forward(self, x):
        x = self.regression_layer(x)
        mean = self.mean_head(x)
        variance = self.variance_head(x)
        return mean, variance

def train_model(model, x, y, fase_1=True, epochs=100, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    criterion_2 = nn.GaussianNLLLoss()


    if fase_1:
        for param in model.variance_head.parameters():
            param.requires_grad = False

    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        mean, variance = model(x_tensor)

        if fase_1:
            loss = criterion(mean, y_tensor)
        else:
            loss = criterion_2(mean, y_tensor, variance)

        loss.backward()
        optimizer.step()
        print( f"Epoch {epoch}, Loss: {loss.item()}")

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

def dropout_inference(model, x , T=50):
    model.eval()

    def enable_dropout(model):
        for m in model.modules():
            if isinstance(m, drop_connect):
                m.force_drop = True

    enable_dropout(model)
    mean_predictions = torch.stack([model(x)[0] for _ in range(T)])
    variance_predictions = torch.stack([model(x)[1] for _ in range(T)])
    return disentangle(mean_predictions, variance_predictions)


def main():
    # Does the model learn the variance or the mean, cause otherwise we need to take the exp of the variance output?
    #  I assume the model is supposed to learn the distribution of the data?

    np.random.seed(42)
    torch.manual_seed(42)

    x = np.linspace(-10, 10, 1000)
    y = sample_distribution(x)

    model = regression_model()
    model = train_model(model, x, y, fase_1=True, epochs=100)
    model = train_model(model, x, y, fase_1=False, epochs=100)

    OOD = np.linspace(-20, 20, 1000)

    epistemic, aleatoric, total_uncertainty = dropout_inference(model, torch.tensor(OOD, dtype=torch.float32).view(-1, 1))

    """Make plots for both predictive
    and each source of uncertainty, and argue/describe/analyze/discuss if each uncertainty estimate makes
    sense or not. For this use your in-distribution and out of distribution datasets"""
    #To see if it performs well we can concatonate the OOD and ID data, create grond truth labels and plot the AUROC which should be high -> SkLearn has a build in AUROC function for this
    #Furthermore we can plot the graphs for both uncertainties for OOD and ID and see the differfence


if __name__ == '__main__':
    main()
