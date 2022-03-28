import os
from posixpath import basename

import gym
import numpy as np
import stable_baselines3 as sb
import torch
from torch import nn
from torch.optim import Adam
from trivial_torch_tools import to_tensor, Sequential, init, convert_each_arg
from trivial_torch_tools.generics import to_pure

from mlp import mlp
from info import path_to, config
from main.training.train_agent import Agent
from main.tools import flatten, get_discounted_rewards, divide_chunks, minibatch, ft, Episode, train_test_split

minibatch_size = config.train_dynamics.minibatch_size

# State transition dynamics model
class DynamicsModel(nn.Module):
    """
    The model of how the world works
        (state) => (next_state)
    """
    
    # 
    # load
    # 
    @classmethod
    def load_default_for(cls, env_name, *, load_previous_weights=True, agent_load_previous_weights=True):
        env = config.get_env(env_name)
        agent = Agent.load_default_for(env_name, load_previous_weights=agent_load_previous_weights)
        dynamics = DynamicsModel(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            hidden_sizes=[64, 64, 64, 64],
            lr=0.0001,
            agent=agent,
            device=config.device,
        )
        if load_previous_weights:
            dynamics.load_state_dict(torch.load(path_to.dynamics_model_for(env_name)))
        return dynamics
    
    # init
    @init.save_and_load_methods(model_attributes=["model"], basic_attributes=[ "hidden_sizes", "learning_rate", "obs_dim", "act_dim"])
    def __init__(self, obs_dim, act_dim, hidden_sizes, lr, device, agent, **kwargs):
        super().__init__()
        self.learning_rate = lr
        self.device        = device
        self.obs_dim       = obs_dim
        self.act_dim       = act_dim
        self.agent         = agent
        self.model = mlp([obs_dim + act_dim, *hidden_sizes, obs_dim], nn.ReLU).to(self.device)
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
    def testing_loss(self, state_batch: torch.Tensor, action_batch: torch.Tensor, next_state_batch: torch.Tensor):
        self.eval() # testing mode
        loss_function = getattr(self, config.train_dynamics.loss_function)
        loss = loss_function(state_batch, action_batch, next_state_batch)
        return to_pure(loss)
        
    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="device")
    def training_loss(self, state_batch: torch.Tensor, action_batch: torch.Tensor, next_state_batch: torch.Tensor):
        self.train() # training mode
        loss_function = getattr(self, config.train_dynamics.loss_function)
        loss = loss_function(state_batch, action_batch, next_state_batch)
        
        # Optimize the self model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    # 
    # Loss function options
    # 
    
    def value_prediction_loss(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        predicted_next_state   = self.predict(state, action)
        
        predicted_next_action = self.agent.make_decision(predicted_next_state, deterministic=True)
        predicted_next_value  = self.agent.value_of(next_state, predicted_next_action)
        best_next_action = self.agent.make_decision(next_state, deterministic=True)
        best_next_value  = self.agent.value_of(next_state, best_next_action)
        
        return (best_next_value - predicted_next_value).mean() # when predicted_next_value is high, loss is low (negative)
    
    def action_prediction_loss(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        predicted_next_state   = self.predict(state, action)
        
        predicted_next_action = self.agent.make_decision(predicted_next_state, deterministic=True)
        best_next_action = self.agent.make_decision(next_state, deterministic=True)
        
        return ((best_next_action - predicted_next_action) ** 2).mean() # when action is very different, loss is high
        
    def state_prediction_loss(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        predicted_next_state = self.predict(state, action)
        
        actual = predicted_next_state
        expected = next_state
        
        return ((actual - expected) ** 2).mean()

def experience(env, agent, n_episodes):
    episodes = [None]*n_episodes
    all_actions, all_curr_states, all_next_states = [], [], []
    print(f'''Starting Experience Recording''')
    for episode_index in range(n_episodes):
        episode = Episode()
        
        state = env.reset()
        episode.states.append(state)
        episode.reward_total = 0
        
        done = False
        while not done:
            action, _ = agent.predict(state, deterministic=True)  # False?
            action = np.random.multivariate_normal(action, 0 * np.identity(len(action))) # QUESTION: why sample from multivariate_normal?
            episode.actions.append(action)
            state, reward, done, info = env.step(action)
            episode.states.append(state)
            episode.reward_total += reward
        
        print(f"    Episode: {episode_index}, Reward: {episode.reward_total:.3f}")
        episodes[episode_index] = episode
        all_curr_states += episode.curr_states
        all_actions     += episode.actions
        all_next_states += episode.next_states

    return episodes, all_actions, all_curr_states, all_next_states

def train(env_name, n_episodes=100, n_epochs=100):
    dynamics = DynamicsModel.load_default_for(env_name, load_previous_weights = False)
    agent    = dynamics.agent
    env      = config.get_env(env_name)

    # Get experience from trained agent
    episodes, all_actions, all_curr_states, all_next_states = experience(env, agent, n_episodes)
    print(f'''Starting Train/Test''')
    states      = to_tensor(all_curr_states)
    actions     = to_tensor(all_actions)
    next_states = to_tensor(all_next_states)

    (
        (train_states     , test_states     ),
        (train_actions    , test_actions    ),
        (train_next_states, test_next_states),
    ) = train_test_split(
        states,
        actions,
        next_states,
        split_proportion=config.train_dynamics.train_test_split,
    )
    
    for epochs_index in range(n_epochs):
        # 
        # training
        # 
        train_losses = []
        for state_batch, action_batch, next_state_batch in minibatch(minibatch_size, train_states, train_actions, train_next_states):
            train_losses.append(dynamics.training_loss(state_batch, action_batch, next_state_batch))
        
        # 
        # testing
        # 
        test_loss = dynamics.testing_loss(test_states, test_actions, test_next_states)
        
        print(f"    Epoch {epochs_index+1}. Train Loss: {to_tensor(train_losses).mean():.4f}, Test Loss: {test_loss:.4f}")

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
