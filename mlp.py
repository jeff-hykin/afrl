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
    """
    The model of how the world works
        (state) => (next_state)
    """
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
        with torch.no_grad(): # QUESTION: this seems really strange, is a different forward-like method called when training the DynamicsModel?
            next_observation = self.model(torch.cat((obs, act), -1))
        return next_observation.cpu().numpy()

    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="device")
    def predict(self, observations: torch.Tensor, actions: torch.Tensor):
        # used for batch predictions
        # expecting observations and actions to be on device
        # returns the predictions still on the device
        return self.model(torch.cat((observations, actions), -1).to(self.device))

    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="device")
    def loss_function(self, actual: torch.Tensor, expected: torch.Tensor):
        # BOOKMARK: loss
        # Compute Huber loss (less sensitive to outliers) # QUESTION: this doesnt look like Huber loss to me
        return ((actual - expected) ** 2).mean()

    
    def apply_loss(dynamics, agent, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        state      = state.to(config.device)
        action     = action.to(config.device)
        next_state = next_state.to(config.device)
        
        predicted_next_state   = dynamics.predict(state, action)
        agent.freeze()
        predicted_next_action = agent.make_decision(predicted_next_state, deterministic=True)
        predicted_next_value  = agent.value_of(next_state, predicted_next_action)
        best_next_action = agent.make_decision(next_state, deterministic=True)
        best_next_value  = agent.value_of(next_state, best_next_action)
        
        print(f'''best_next_value = {best_next_value}''')
        print(f'''predicted_next_value = {predicted_next_value}''')
        loss = (best_next_value - predicted_next_value).mean() # when predicted_next_value is high, loss is low (negative)
        
        # Optimize the dynamics model
        dynamics.optimizer.zero_grad()
        loss.backward()
        dynamics.optimizer.step()
        
        agent.unfreeze()

        return loss


class RobustPredictivePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, lr, hidden_sizes):
        super().__init__()
        self.model = mlp([obs_dim, *hidden_sizes, act_dim], nn.ReLU)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def predict(self, obs: torch.Tensor):
        return self.model(obs)

    def forward(self, obs):
        return self.predict(obs)
