import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import Config


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.input_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.output_dim = 1

        self.truncation_range = config.truncation_range

        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(self.input_dim, self.hidden_dim, True))
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, True))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(self.hidden_dim, self.output_dim, True)

        self.to(config.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(input))
            else:
                h = F.relu(l(h))
                
        output = self.lout(h).squeeze(-1) * self.truncation_range

        return output