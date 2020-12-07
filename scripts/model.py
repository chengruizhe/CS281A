import torch
import torch.nn as nn
import torch.nn.functional as F


class mlp(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(11, 11) for i in range(self.layers)])
        self.classifier = nn.Linear(11, 1)

    def forward(self, x):
        for i in range(self.layers):
            x = self.linears[i](x)
            x = F.relu(x)
        x = self.classifier(x)
        x = F.sigmoid(x)
        return x