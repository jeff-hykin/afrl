import numpy as np
import torch as th
from torch import FloatTensor as ft
from torch import nn
from torch.optim import Adam


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class DynamicsModel(nn.Module):
    # State transition dynamics model
    def __init__(self, obs_dim, act_dim, hidden_sizes, lr, device):
        super().__init__()
        self.device = device
        self.model = mlp([obs_dim + act_dim, *hidden_sizes, obs_dim], nn.ReLU).to(
            self.device
        )
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def forward(self, obs: np.ndarray, act: np.ndarray):
        # used for a single state and action
        obs = ft(obs).to(self.device)
        act = ft(act).to(self.device)
        with th.no_grad():
            next_observation = self.model(th.cat((obs, act), -1))
        return next_observation.cpu().numpy()

    def predict(self, obs: th.Tensor, act: th.Tensor):
        # used for batch predictions
        # expecting obs and act to be on device
        # returns the predictions still on the device
        return self.model(th.cat((obs, act), -1))


class RobustPredictivePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, lr, hidden_sizes):
        super().__init__()
        self.model = mlp([obs_dim, *hidden_sizes, act_dim], nn.ReLU)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def predict(self, obs: th.Tensor):
        return self.model(obs)

    def forward(self, obs):
        return self.predict(obs)
