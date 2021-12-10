import os
from copy import deepcopy

import gym
import numpy as np
import pandas as pd
import scipy.stats
import stable_baselines3 as sb
import torch
from torch import FloatTensor as ft
from torch.optim.adam import Adam
from tqdm import tqdm

from dynamics import DynamicsModel

env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)

dynamics = DynamicsModel(
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.shape[0],
    hidden_sizes=[64, 64],
    lr=0.0001,
    device='cpu')


def load_agent(env):
    agent_base_path = 'data/models/agents'
    agent_path = os.path.join(agent_base_path, env + '.zip')
    return sb.SAC.load(agent_path, gym.make(env_name), device='cpu')


agent = load_agent(env_name)


# todo: Seed with SAC policy
predpolicy = deepcopy(agent.policy)
optimizer = Adam(predpolicy.parameters(), lr=0.0001)

dynamics.load_state_dict(torch.load(f'data/models/dynamics/{env_name}.pt'))


def Q(agent, state: np.ndarray, action: np.ndarray):
    state = ft([state]).to(agent.device)
    if not torch.is_tensor(action):
        action = ft([action]).to(agent.device)
    # with torch.no_grad():
    q = torch.cat(agent.critic_target(state, action), dim=1)
    q, _ = torch.min(q, dim=1, keepdim=True)
    return q


def predict_future_state(state, horizon):
    for _ in range(horizon):
        action = agent.predict(state, deterministic=False)[0]
        state = dynamics.forward(state, action)
    return state


def train(horizon, n_episodes):
    pred_states = []
    ep_loss = []
    losses = []
    for i in tqdm(list(range(n_episodes))):
        done = False
        state = env.reset()

        while not done:
            action = agent.predict(state, deterministic=True)[0]
            state, reward, done, _ = env.step(action)
            pred_s = predict_future_state(state, horizon)
            pred_states.append(pred_s)
            if len(pred_states) >= horizon:
                loss = Q(agent, state, action) \
                    - Q(agent, state, predpolicy(ft([pred_states[-horizon]])))
                losses.append(loss)
                if len(losses) == 32:
                    # Optimize the predpolicy
                    optimizer.zero_grad()
                    loss = torch.stack(losses).mean()
                    loss.backward()
                    optimizer.step()
                    losses = []
                    ep_loss.append(round(loss.item(), 2))

        if i % 50 == 0:
            # torch.save(predpolicy.state_dict(), f'data/models/agents/predpolicy/{env_name}.pt')
            print(i, round(np.mean(ep_loss[:50]), 3), round(
                np.mean(ep_loss[-50:]), 3))

            eq_mean_test = scipy.stats.ttest_ind(
                ep_loss[:50], ep_loss[-50:], axis=0, equal_var=False)
            print(eq_mean_test)

    return ep_loss


def test(horizon):
    pred_states = []
    done = False
    while not done:
        if len(pred_states) >= horizon:
            action = predpolicy(pred_states[-horizon])
        else:
            action = agent.predict(state, deterministic=False)[0]
        state, reward, done, _ = env.step(action)
        pred_s = predict_future_state(state, horizon)
        pred_states.append(pred_s)


losses = train(horizon=5, n_episodes=5000)
