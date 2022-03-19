import os
from posixpath import basename

import gym
import numpy as np
import stable_baselines3 as sb
import torch
from torch import nn
from torch import FloatTensor as ft
from torch.optim import Adam
from trivial_torch_tools import Sequential, init, convert_each_arg

from mlp import mlp
from train_agent import load_agent
from info import path_to, config

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

    def coach_loss(dynamics, agent, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        predicted_next_state   = dynamics.predict(state, action)
        predicted_next_action = agent.make_decision(predicted_next_state, deterministic=True)
        predicted_next_value  = agent.value_of(next_state, predicted_next_action)
        best_next_action = agent.make_decision(next_state, deterministic=True)
        best_next_value  = agent.value_of(next_state, best_next_action)
        
        return (best_next_value - predicted_next_value).mean() # when predicted_next_value is high, loss is low (negative)
        
    def mse_loss(dynamics, agent, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        predicted_next_state = dynamics.predict(state, action)
        predicted_action, _  = agent.predict(predicted_next_state, deterministic=True)
        action, _            = agent.predict(next_state, deterministic=True)
        
        actual = agent.value_of(next_state, predicted_action)
        expected = agent.value_of(next_state, action)
        return ((actual - expected) ** 2).mean()
    
    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="device")
    def test_loss(self, actual, expected):
        return ((actual - expected) ** 2).mean()
        
    def apply_loss(self, agent, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        state      = state.to(config.device)
        action     = action.to(config.device)
        next_state = next_state.to(config.device)
        self.train()
        
        loss = self.mse_loss(agent, state, action, next_state)
            
        # Optimize the dynamics model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss




def load_dynamics(env_obj):
    env = env_obj
    return DynamicsModel(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        hidden_sizes=[64, 64, 64, 64],
        lr=0.0001,
        device=config.device,
    )


def flatten(ys):
    return [x for xs in ys for x in xs]


def get_discounted_rewards(gamma, rewards):
    return sum([r * gamma ** t for t, r in enumerate(rewards)])


def experience(env, agent, n_episodes):
    actions, states = [], []
    for i in range(n_episodes):
        done = False
        obs = env.reset()
        ep_actions, ep_states = [], []
        ep_states.append(obs)
        ep_reward = 0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)  # False?
            action = np.random.multivariate_normal(action, 0 * np.identity(len(action))) # QUESTION: why sample from multivariate_normal?
            ep_actions.append(action)
            obs, reward, done, info = env.step(action)
            if i == 0:
                # env.render()
                pass
            ep_states.append(obs)
            ep_reward += reward
        print(f"Episode: {i}, Reward: {ep_reward:.3f}")
        states.append(ep_states)
        actions.append(ep_actions)

    return states, actions


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def minibatch(batch_size, *data):
    indices = np.arange(len(data[0]))
    np.random.shuffle(indices)
    for batch_ind in divide_chunks(indices, batch_size):
        yield [datum[batch_ind] for datum in data]


def train(env_name, n_episodes=100, n_epochs=100):
    env = config.get_env(env_name)
    agent = load_agent(env_name)

    # Get experience from trained agent
    states, actions = experience(env, agent, n_episodes)

    next_states = torch.FloatTensor(
        flatten([[s for s in ep_states[1:]] for ep_states in states])
    )
    states = torch.FloatTensor(
        flatten([[s for s in ep_states[:-1]] for ep_states in states])
    )
    actions = torch.FloatTensor(flatten(actions))

    dynamics = load_dynamics(env)

    def train_test_split(data, indices, train_pct=0.66):
        div = int(len(data) * train_pct)
        train, test = indices[:div], indices[div:]
        return data[train], data[test]

    indices = np.arange(len(states))
    np.random.shuffle(indices)
    train_s, test_s = train_test_split(states, indices, config.train_dynamics.train_test_split)
    train_a, test_a = train_test_split(actions, indices, config.train_dynamics.train_test_split)
    train_s2, test_s2 = train_test_split(next_states, indices, config.train_dynamics.train_test_split)
    
    minibatch_size = config.train_dynamics.minibatch_size
    for epochs_index in range(n_epochs):
        loss = 0
        for s, a, s2 in minibatch(minibatch_size, train_s, train_a, train_s2):
            batch_loss = dynamics.apply_loss(agent, s, a, s2)
            loss += batch_loss
            
        test_mse = dynamics.test_loss(
            actual=dynamics.predict(test_s, test_a),
            expected=test_s2,
        )
        print(
            f"Epoch {epochs_index+1}. Loss: {loss / np.ceil(len(states) / minibatch_size):.4f}, Test mse: {test_mse:.4f}"
        )

    torch.save(dynamics.state_dict(), path_to.dynamics_model_for(env_name))


if __name__ == '__main__':
    for each_env_name in config.env_names:
        print(f"")
        print(f"")
        print(f"Training for {each_env_name}")
        print(f"")
        print(f"")
        train(
            each_env_name,
            n_episodes=config.train_dynamics.number_of_episodes,
            n_epochs=config.train_dynamics.number_of_epochs,
        )
