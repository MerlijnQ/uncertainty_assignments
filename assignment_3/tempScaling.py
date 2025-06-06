import torch
from torch import nn, optim
from torch.nn import functional as F

"We scale the temperature to get a better calibrated model. The code is inspired on the following github: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py#L78"


class CalibratedModel(nn.Module):
    def __init__(self, model, device='cpu'):
        super(CalibratedModel, self).__init__()
        self.model = model.eval()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5).to(device)

    def forward(self, input):
        logits = self.model(input)
        if isinstance(logits, tuple):
            logits, _ = logits
        temp = self.temperature.unsqueeze(1).expand_as(logits)
        return logits/temp


class Calibrate():
    def __init__(self, criterion=nn.CrossEntropyLoss, device='cpu'):  
        self.criterion = criterion
        self.device = device

    
    def optimize(self, val_loader, model):

        model.eval()
        model = CalibratedModel(model, device=self.device)

        all_inputs, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                all_inputs.append(inputs)
                all_labels.append(labels)

        all_inputs = torch.cat(all_inputs).to(self.device)
        all_labels = torch.cat(all_labels).to(self.device)

        optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            logits = model(all_inputs)
            loss = self.criterion(logits, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        print(f"Optimal temperature: {model.temperature.item():.4f}")
        return model
