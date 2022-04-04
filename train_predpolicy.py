import os
from copy import deepcopy

import gym
import numpy as np
import pandas as pd
import scipy.stats
import stable_baselines3 as sb
import torch
from torch.optim.adam import Adam
from tqdm import tqdm

from info import path_to, config
from main.training.train_agent import Agent
from main.training.train_dynamics import DynamicsModel
from main.tools import flatten, get_discounted_rewards, divide_chunks, minibatch, ft

env_name = "LunarLanderContinuous-v2"

dynamics = DynamicsModel.load_default_for(env_name)
agent    = dynamics.agent
env      = config.get_env(env_name)

# todo: Seed with SAC policy
predpolicy = deepcopy(agent.policy)
optimizer = Adam(predpolicy.parameters(), lr=0.0001)


def predict_future_state(state, horizon):
    for _ in range(horizon):
        # BOOKMARK: why the new loss does poorly
        action = agent.predict(state, deterministic=False)[0]
        state = dynamics.predict(state, action)
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
                loss = agent.value_of(state, action) - agent.value_of(state, predpolicy(ft([pred_states[-horizon]])))
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
            print(i, round(np.mean(ep_loss[:50]), 3), round(np.mean(ep_loss[-50:]), 3))

            eq_mean_test = scipy.stats.ttest_ind(
                ep_loss[:50], ep_loss[-50:], axis=0, equal_var=False
            )
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
