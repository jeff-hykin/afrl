import numpy as np
import torch as torch
from torch import nn
from torch.optim import Adam
from trivial_torch_tools import Sequential, init, convert_each_arg

from info import config


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class RobustPredictivePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, lr, hidden_sizes):
        super().__init__()
        self.model = mlp([obs_dim, *hidden_sizes, act_dim], nn.ReLU)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def predict(self, obs: torch.Tensor):
        return self.model(obs)

    def forward(self, obs):
        return self.predict(obs)
