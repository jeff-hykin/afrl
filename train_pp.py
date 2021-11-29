import os

import gym
import numpy as np
import pandas as pd
import stable_baselines3 as sb
import torch
from torch import FloatTensor as ft
from tqdm import tqdm

from dynamics import DynamicsModel, RobustPredictivePolicy

env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)

dynamics = DynamicsModel(
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.shape[0],
    hidden_sizes=[64, 64, 64, 64],
    lr=0.0001,
    device='cpu')

predpolicy = RobustPredictivePolicy(
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.shape[0],
    hidden_sizes=[64, 64, 64, 64],
    lr=0.0001)

dynamics.load_state_dict(torch.load(f'data/models/dynamics/{env_name}.pt'))


def load_agent(env):
    agent_base_path = 'data/models/agents'
    agent_path = os.path.join(agent_base_path, env + '.zip')
    return sb.SAC.load(agent_path, gym.make(env_name), device='cpu')


agent = load_agent(env_name)


def train_pred_model(dynamics, agent, state: torch.Tensor, action: torch.Tensor, pred_s: torch.Tensor):
    # Compute Huber loss (less sensitive to outliers)
    loss = Q(agent, state, action) - \
        Q(agent, state, predpolicy.predict(ft([pred_s])))
    # Optimize the dynamics model
    dynamics.optimizer.zero_grad()
    loss.backward()
    dynamics.optimizer.step()

    return loss.item()


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


def train(horizon):
  pred_states = []
  ep_loss = []
  n_episodes = 200
  for _ in tqdm(list(range(n_episodes))):
      losses = []
      done = False
      state = env.reset()

      while not done:
          action = agent.predict(state, deterministic=False)[0]
          state, reward, done, _ = env.step(action)
          pred_s = predict_future_state(state, horizon)
          pred_states.append(pred_s)
          if len(pred_states) >= horizon:
              loss = train_pred_model(
                  dynamics, agent, state, action, pred_states[-H])
              losses.append(loss)
      ep_loss.append(round(sum(losses), 2))

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
