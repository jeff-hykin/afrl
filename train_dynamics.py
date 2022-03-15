import os
from posixpath import basename

import gym
import numpy as np
import stable_baselines3 as sb
import torch

from mlp import DynamicsModel
from train_agent import load_agent
from info import path_to, config

def load_dynamics(env_obj):
    env = env_obj
    return DynamicsModel(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        hidden_sizes=[64, 64, 64, 64],
        lr=0.0001,
        device=config.device,
    )


def train_dynamics_model(dynamics, agent, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
    from train_afrl_pp import Q
    
    state = state.to(config.device)
    action = action.to(config.device)
    next_state = next_state.to(config.device)
    
    predicted_next_state = dynamics.predict(state, action)
    predicted_action, _  = agent.predict(predicted_next_state, deterministic=True)
    action, _            = agent.predict(next_state, deterministic=True)
    
    loss = dynamics.loss_function(
        actual=Q(next_state, predicted_action),
        expected=Q(next_state, action),
    )
    
    # Optimize the dynamics model
    dynamics.optimizer.zero_grad()
    loss.backward()
    dynamics.optimizer.step()

    return loss


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
            action = np.random.multivariate_normal(action, 0 * np.identity(len(action)))
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
    for i in range(n_epochs):
        loss = 0
        for s, a, s2 in minibatch(minibatch_size, train_s, train_a, train_s2):
            batch_loss = train_dynamics_model(dynamics, agent, s, a, s2)
            loss += batch_loss
            
        test_mse = dynamics.loss_function(
            actual=dynamics.predict(test_s, test_a),
            expected=test_s2,
        )
        print(
            f"Epoch {i+1}. Loss: {loss / np.ceil(len(states) / minibatch_size):.4f}, Test mse: {test_mse:.4f}"
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
