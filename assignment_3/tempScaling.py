import torch
from torch import nn, optim
from torch.nn import functional as F

"We scale the temperature to get a better calibrated model. The code is inspired on the following github: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py#L78"

class calibratedModel(nn.Module):
    def __init__(self, model):
        super(calibratedModel, self).__init__()
        self.model = model.eval()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        temp = self.temperature.unsqueeze(1).expand_as(logits)
        return logits/temp
    
class calibrate():
    def __init__(self, model: calibratedModel, criterion=nn.CrossEntropyLoss, device='cpu'):  
        self.model = model
        self.criterion = criterion
        self.device = device

    
    def optimize(self, val_loader):

        all_inputs, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                all_inputs.append(inputs)
                all_labels.append(labels)

        all_inputs = torch.cat(all_inputs)
        all_labels = torch.cat(all_labels)

        optimizer = optim.LBFGS([self.model.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            logits = self.model(all_inputs)
            loss = self.criterion(logits, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        print(f"Optimal temperature: {self.model.temperature.item():.4f}")
        return self.model        

        
