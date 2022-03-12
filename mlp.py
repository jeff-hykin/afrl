import numpy as np
import torch as torch
from torch import FloatTensor as ft
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

# State transition dynamics model
class DynamicsModel(nn.Module):
    @init.save_and_load_methods(model_attributes=["model"], basic_attributes=["learning_rate"])
    def __init__(self, obs_dim, act_dim, hidden_sizes, lr, device, **kwargs):
        super().__init__()
        self.learning_rate = lr
        self.device = device
        self.model = mlp([obs_dim + act_dim, *hidden_sizes, obs_dim], nn.ReLU).to(
            self.device
        )
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="device")
    def forward(self, obs: np.ndarray, act: np.ndarray):
        with torch.no_grad():
            next_observation = self.model(torch.cat((obs, act), -1))
        return next_observation.cpu().numpy()

    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="device")
    def predict(self, obs: torch.Tensor, act: torch.Tensor):
        # used for batch predictions
        # expecting obs and act to be on device
        # returns the predictions still on the device
        return self.model(torch.cat((obs, act), -1).to(self.device))

    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="device")
    def loss_function(self, actual: torch.Tensor, expected: torch.Tensor):
        # BOOKMARK: loss
        # Compute Huber loss (less sensitive to outliers) # QUESTION: this doesnt look like Huber loss to me
        return ((actual - expected) ** 2).mean()
        

class RobustPredictivePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, lr, hidden_sizes):
        super().__init__()
        self.model = mlp([obs_dim, *hidden_sizes, act_dim], nn.ReLU)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def predict(self, obs: torch.Tensor):
        return self.model(obs)

    def forward(self, obs):
        return self.predict(obs)
