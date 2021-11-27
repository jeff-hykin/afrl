import os
from types import FunctionType
from typing import List

import gym
import numpy as np
import pandas as pd
import stable_baselines3 as sb
import torch
from nptyping import NDArray
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from torch import FloatTensor as ft
from tqdm import tqdm

from dynamics import DynamicsModel


def get_env(env_name):
    env = gym.make(env_name)
    return env


def Q(agent, state: np.ndarray, action: np.ndarray):
    state = ft([state]).to(agent.device)
    action = ft([action]).to(agent.device)
    with torch.no_grad():
        q = torch.cat(agent.critic_target(state, action), dim=1)
    q, _ = torch.min(q, dim=1, keepdim=True)
    return q.item()


def load_agent(env):
    if 'Humanoid' in env:
        agent_path = 'log/best_model.zip'
    else:
        agent_base_path = 'data/models/agents'
        agent_path = os.path.join(agent_base_path, env + '.zip')
    return sb.SAC.load(agent_path, get_env(env), device='cpu')


def get_dynamics_path(env):
    base_path = 'data/models/dynamics'
    return os.path.join(base_path, env + '.pt')


def replan(state: NDArray, old_plan: NDArray,
           forecast: List[int], delta: float,
           forecast_horizon: int, action_size: int,
           agent: OffPolicyAlgorithm, dynamics: DynamicsModel):

    new_plan = np.empty((forecast_horizon, action_size), dtype=np.float)
    k = 0  # used to keep track of forecast of the actions

    # reuse old plan (recycle)
    for action in old_plan[1:]:
        with torch.no_grad():
            replan_action = agent.predict(state, deterministic=True)[0]
            replan_q = Q(agent, state, replan_action)
            plan_q = Q(agent, state, action)
            if plan_q + delta < replan_q:
                break
            new_plan[k] = action
            state = dynamics(state, action)
        # for the stats... keep track of the forecast of this action
        forecast[k] = forecast[k+1] + 1
        k += 1

    # produce new plan (replan)
    for i in range(k, forecast_horizon):
        action = agent.predict(state, deterministic=True)[0]
        new_plan[i] = action
        with torch.no_grad():
            state = dynamics(state, action)
        forecast[i] = 0

    return new_plan, forecast


def experience(delta: float, forecast_horizon: int, action_size: int,
               agent: OffPolicyAlgorithm, dynamics: DynamicsModel, env):
    episode_forecast = []
    rewards = []
    empty_plan = []
    zero_forecasts = np.zeros(forecast_horizon, np.int8)
    state = env.reset()
    done = False
    plan, forecasts = replan(
        state, empty_plan, zero_forecasts, delta,
        forecast_horizon, action_size, agent, dynamics)
    while not done:
        action = plan[0]
        episode_forecast.append(forecasts[0])
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        plan, forecasts = replan(
            state, plan, forecasts, delta,
            forecast_horizon, action_size, agent, dynamics)
    return rewards, episode_forecast


def get_discounted_rewards(rewards, gamma):
    return sum([r * gamma**t for t, r in enumerate(rewards)])


def test_afrl(deltas: List[float], forecast_horizon: int, action_size: int,
              agent: OffPolicyAlgorithm, dynamics: DynamicsModel,
              n_experiments: int, env):
    cols = ['delta', 'rewards', 'discounted_rewards', 'forecast']
    if isinstance(forecast_horizon, list):
        assert len(forecast_horizon) == len(deltas)
    else:
        forecast_horizon = [forecast_horizon]*len(deltas)
    df = pd.DataFrame(columns=cols)
    for delta, horizon in zip(deltas, forecast_horizon):
        for _ in tqdm(range(n_experiments)):
            rewards, forecast = experience(
                delta, horizon, action_size,
                agent, dynamics, env)
            v = get_discounted_rewards(rewards, agent.gamma)
            df = df.append(
                dict(zip(cols, [delta, sum(rewards), v, forecast[horizon:]])), ignore_index=True)

    return df


def main(env_name, n_experiments=1, forecast_horizon=1, deltas=[0]):
    env = get_env(env_name)
    dynamics = DynamicsModel(obs_dim=env.observation_space.shape[0],
                             act_dim=env.action_space.shape[0],
                             hidden_sizes=[64,64,64,64],
                             lr=0.0001,
                             device='cpu')

    dynamics.load_state_dict(torch.load(
        get_dynamics_path(env_name)))
    action_size = env.action_space.shape[0]
    agent = load_agent(env_name)
    # agent.gamma = 0.95
    print('Gamma:', agent.gamma)

    return test_afrl(
        deltas, forecast_horizon, action_size, agent,
        dynamics, n_experiments, env)


def get_results_folder():
    results_folder = os.path.join(f'data/results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    return results_folder


settings = {
    'LunarLanderContinuous-v2': {
        'max_score': 70, # (discounted?) ep reward
        'min_score': -100,
        'horizons': {
            0.001: 3, 
            0.0025: 5, 
            0.005: 10, 
            0.0075: 15, 
            0.01: 20
        }
    }
}

if __name__ == '__main__':
    env = 'LunarLanderContinuous-v2'
    multipliers = np.array([0.001, 0.0025, 0.005, 0.0075, 0.01])
    deltas = (settings[env]['max_score'] - settings[env]['min_score']) * multipliers
    horizons = [settings[env]['horizons'][mult] for mult in multipliers]
    df = main(env, 20, horizons, deltas=deltas)
    df = df.explode('forecast')
    print(df)
    results_folder = get_results_folder()
    df.to_csv(f'{results_folder}/{env}/experiments.csv')
    # print(df.groupby('delta').forecast.mean())
    # print((df.groupby('delta').discounted_rewards.mean() - envs[env]['min']) / (envs[env]['max'] - envs[env]['min']))