import torch
from torch import nn

class CNNGenre(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0   