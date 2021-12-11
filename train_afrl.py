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

from mlp import DynamicsModel
from torch.optim.adam import Adam


def Q(agent, state: np.ndarray, action: np.ndarray):
    if torch.is_tensor(action):
        action = torch.unsqueeze(action, 0)
    else:
        action = ft([action])
    q = torch.cat(agent.critic_target(ft([state]), action), dim=1)
    q, _ = torch.min(q, dim=1, keepdim=True)
    return q


def load_agent(env):
    if 'Humanoid' in env:
        agent_path = 'log/best_model.zip'
    else:
        agent_base_path = 'data/models/agents'
        agent_path = os.path.join(agent_base_path, env + '.zip')
    return sb.SAC.load(agent_path, gym.make(env), device='cpu')


def get_dynamics_path(env):
    base_path = 'data/models/dynamics'
    return os.path.join(base_path, env + '.pt')


losses = []

def replan(state: NDArray, old_plan: NDArray,
           forecast: List[int], epsilon: float,
           forecast_horizon: int, action_size: int,
           agent: OffPolicyAlgorithm, predpolicy, dynamics: DynamicsModel, optimizer):
    global losses

    new_plan = []
    k = 0  # used to keep track of forecast of the actions

    # reuse old plan (recycle)
    for action in old_plan[1:]:
        replan_action = agent.predict(state, deterministic=True)[0]
        with torch.no_grad():
          replan_q = Q(agent, state, replan_action)
        plan_action = predpolicy(ft([state]))[0]
        plan_q = Q(agent, state, plan_action)
        diff = replan_q - plan_q
        losses.append(diff)
        if len(losses) == 32:
            optimizer.zero_grad()
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            losses = []

        if diff.item() > epsilon:
            break
        new_plan.append(plan_action)
        state = dynamics(state, plan_action)
        # for the stats... keep track of the forecast of this action
        forecast[k] = forecast[k+1] + 1
        k += 1

    # produce new plan (replan)
    for i in range(k, forecast_horizon):
        action = predpolicy(ft([state]))[0]
        new_plan.append(action)
        with torch.no_grad():
            state = dynamics(state, action)
        forecast[i] = 0

    return new_plan, forecast


def experience(epsilon: float, forecast_horizon: int, action_size: int,
               agent: OffPolicyAlgorithm, predpolicy, dynamics: DynamicsModel, env, optimizer):
    episode_forecast = []
    rewards = []
    empty_plan = []
    zero_forecasts = np.zeros(forecast_horizon, np.int8)
    state = env.reset()
    done = False
    plan, forecasts = replan(
        state, empty_plan, zero_forecasts, epsilon,
        forecast_horizon, action_size, agent, predpolicy, dynamics, optimizer)
    while not done:
        action = plan[0]
        episode_forecast.append(forecasts[0])
        state, reward, done, _ = env.step(action.detach().numpy())
        rewards.append(reward)
        plan, forecasts = replan(
            state, plan, forecasts, epsilon,
            forecast_horizon, action_size, agent, predpolicy, dynamics, optimizer)
    return rewards, episode_forecast


def get_discounted_rewards(rewards, gamma):
    return sum([r * gamma**t for t, r in enumerate(rewards)])


def test_afrl(epsilons: List[float], forecast_horizon: int, action_size: int,
              agent: OffPolicyAlgorithm, predpolicy, dynamics: DynamicsModel,
              n_experiments: int, env, optimizer):
    cols = ['epsilon', 'rewards', 'discounted_rewards', 'forecast']
    if isinstance(forecast_horizon, list):
        assert len(forecast_horizon) == len(epsilons)
    else:
        forecast_horizon = [forecast_horizon]*len(epsilons)
    df = pd.DataFrame(columns=cols)
    for epsilon, horizon in zip(epsilons, forecast_horizon):
        for _ in tqdm(range(n_experiments), disable=True):
            rewards, forecast = experience(
                epsilon, horizon, action_size,
                agent, predpolicy, dynamics, env, optimizer)
            v = get_discounted_rewards(rewards, agent.gamma)
            df = df.append(
                dict(zip(cols, [epsilon, sum(rewards), v, forecast[horizon:]])), ignore_index=True)
            print(epsilon, np.mean([np.mean(x) for x in df.forecast]))
    return df


def main(env_name, n_experiments=1, forecast_horizon=1, epsilons=[0]):
    env = gym.make(env_name)
    dynamics = DynamicsModel(obs_dim=env.observation_space.shape[0],
                             act_dim=env.action_space.shape[0],
                             hidden_sizes=[64,64,64,64],
                             lr=0.0001,
                             device='cpu')

    dynamics.load_state_dict(torch.load(
        get_dynamics_path(env_name)))
    action_size = env.action_space.shape[0]
    agent = load_agent(env_name)
    predpolicy = deepcopy(agent.policy)
    optimizer = Adam(predpolicy.parameters(), lr=0.0001)

    # predpolicy.load_state_dict(torch.load(f'data/models/agents/predpolicy/{env_name}.pt'))

    # agent.gamma = 0.95
    print('Gamma:', agent.gamma)

    return test_afrl(
        epsilons, forecast_horizon, action_size, agent, predpolicy,
        dynamics, n_experiments, env, optimizer)


def get_results_folder():
    results_folder = os.path.join(f'data/results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    return results_folder


settings = {
    'LunarLanderContinuous-v2': {
        'max_score': 70, # discounted ep reward
        'min_score': -100,
        'horizons': { # maps "epsilon coefficients" to horizons
            0.001: 5, 
            0.0025: 5, 
            0.005: 10,
            0.0075: 15, 
            0.01: 20
        }
    }
}

if __name__ == '__main__':
    env = 'LunarLanderContinuous-v2'
    from copy import deepcopy

    multipliers = np.array(list(settings[env]['horizons'].keys()))
    epsilons = (settings[env]['max_score'] - settings[env]['min_score']) * multipliers
    horizons = [settings[env]['horizons'][mult] for mult in multipliers]
    df = main(env, 50, horizons, epsilons=epsilons)
    df = df.explode('forecast')
    results_folder = get_results_folder()
    df.to_csv(f'{results_folder}/{env}/experiments_pp.csv')
    # print(df.groupby('epsilon').forecast.mean())
    # print((df.groupby('epsilon').discounted_rewards.mean() - envs[env]['min']) / (envs[env]['max'] - envs[env]['min']))