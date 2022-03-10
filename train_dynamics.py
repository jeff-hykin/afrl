import os
from posixpath import basename

import gym
import numpy as np
import stable_baselines3 as sb
import torch

from mlp import DynamicsModel
from info import path_to, config

def get_env(env_name):
    env = gym.make(env_name)
    return env


def load_agent(env):
    agent_path1 = os.path.join(path_to.folders.agent_models, env)
    agent_path2 = os.path.join(path_to.folders.agent_models, env + '.zip')
    try:
        return sb.SAC.load(agent_path1, get_env(env), device=config.device)
    except Exception as error:
        return sb.SAC.load(agent_path2, get_env(env), device=config.device)


def get_dynamics_path(env):
    return os.path.join(path_to.folders.dynamics_models, env + '.pt')


def train_dynamics_model(dynamics, s: torch.Tensor, a: torch.Tensor, s2: torch.Tensor):
    # Compute Huber loss (less sensitive to outliers)
    loss = ((dynamics.predict(s, a) - s2)**2).mean()
    # Optimize the dynamics model
    dynamics.optimizer.zero_grad()
    loss.backward()
    dynamics.optimizer.step()

    return loss


def flatten(ys): return [x for xs in ys for x in xs]


def get_discounted_rewards(gamma, rewards):
    return sum([r * gamma**t for t, r in enumerate(rewards)])


def experience(env, agent, n_episodes):
    actions, states = [], []
    for i in range(n_episodes):
        done = False
        obs = env.reset()
        ep_actions, ep_states = [], []
        ep_states.append(obs)
        ep_reward = 0
        while not done:
            action, _ = agent.predict(obs, deterministic=True) # False?
            action = np.random.multivariate_normal(action, 0*np.identity(len(action)))
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
        yield l[i:i + n]


def minibatch(batch_size, *data):
    indices = np.arange(len(data[0]))
    np.random.shuffle(indices)
    for batch_ind in divide_chunks(indices, batch_size):
        yield [datum[batch_ind] for datum in data]


def train(env_name, n_episodes=100, n_epochs=100):
    env = get_env(env_name)
    agent = load_agent(env_name)

    # Get experience from trained agent
    states, actions = experience(env, agent, n_episodes)

    next_states = torch.FloatTensor(flatten([[s for s in ep_states[1:]] for ep_states in states]))
    states = torch.FloatTensor(flatten([[s for s in ep_states[:-1]] for ep_states in states]))
    actions = torch.FloatTensor(flatten(actions))

    dynamics = DynamicsModel(obs_dim=env.observation_space.shape[0], 
                             act_dim=env.action_space.shape[0],
                             hidden_sizes=[64,64,64,64],
                             lr=0.0001,
                             device='cpu')

    def train_test_split(data, indices, train_pct=0.66):
        div = int(len(data) * train_pct)
        train, test = indices[:div], indices[div:]
        return data[train], data[test]

    indices = np.arange(len(states))
    np.random.shuffle(indices)
    train_s, test_s = train_test_split(states, indices, 0.7)
    train_a, test_a = train_test_split(actions, indices, 0.7)
    train_s2, test_s2 = train_test_split(next_states, indices, 0.7)

    for i in range(n_epochs):
        loss = 0
        for s, a, s2 in minibatch(32, train_s, train_a, train_s2):
            batch_loss = train_dynamics_model(dynamics, s, a, s2)
            loss += batch_loss
        test_mse = ((dynamics.predict(test_s, test_a) - test_s2)**2).mean()
        print(f'Epoch {i+1}. Loss: {loss / np.ceil(len(states) / 32):.4f}, Test mse: {test_mse:.4f}')

    torch.save(dynamics.state_dict(), get_dynamics_path(env_name))

env_name = 'LunarLanderContinuous-v2'

train(env_name, n_episodes=20, n_epochs=100)
