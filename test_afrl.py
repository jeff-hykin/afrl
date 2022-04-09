import os
from types import FunctionType
from typing import List
from copy import deepcopy
from dataclasses import dataclass

import gym
import numpy as np
import pandas as pd
import stable_baselines3 as sb
import torch
from nptyping import NDArray
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from torch.optim.adam import Adam
from tqdm import tqdm
import file_system_py as FS
from trivial_torch_tools import to_tensor, Sequential, init, convert_each_arg
from trivial_torch_tools.generics import to_pure, flatten
from super_map import LazyDict
from simple_namespace import namespace

from info import path_to, config
from main.training.train_agent import Agent
from main.training.train_dynamics import DynamicsModel
from main.tools import flatten, get_discounted_rewards, divide_chunks, minibatch, ft, TimestepSeries, to_numpy, average

settings = config.gym_env_settings



def replan(
    state: NDArray,
    old_plan: NDArray,
    forecast: List[int],
    epsilon: float,
    forecast_horizon: int,
    action_size: int,
    agent: OffPolicyAlgorithm,
    dynamics: DynamicsModel,
):

    new_plan = np.empty((forecast_horizon, action_size), dtype=np.float)
    k = 0  # used to keep track of forecast of the actions

    # reuse old plan (recycle)
    for action in old_plan[1:]:
        with torch.no_grad():
            replan_action = agent.predict(state, deterministic=True)[0]
            replan_q = agent.value_of(state, replan_action)
            plan_q = agent.value_of(state, action)
            if plan_q + epsilon < replan_q:
                break
            new_plan[k] = action
            state = dynamics(state, action)
        # for the stats... keep track of the forecast of this action
        forecast[k] = forecast[k + 1] + 1
        k += 1

    # produce new plan (replan)
    for i in range(k, forecast_horizon):
        action = agent.predict(state, deterministic=True)[0]
        new_plan[i] = action
        with torch.no_grad():
            state = dynamics(state, action)
        forecast[i] = 0

    return new_plan, forecast


def test_afrl(
    epsilons: List[float],
    forecast_horizon: int,
    action_size: int,
    agent: OffPolicyAlgorithm,
    dynamics: DynamicsModel,
    n_experiments: int,
    env,
):
    cols = ["epsilon", "rewards", "discounted_rewards", "forecast"]
    if isinstance(forecast_horizon, list):
        assert len(forecast_horizon) == len(epsilons)
    else:
        forecast_horizon = [forecast_horizon] * len(epsilons)
    df = pd.DataFrame(columns=cols)
    for epsilon, horizon in zip(epsilons, forecast_horizon):
        for _ in tqdm(range(n_experiments), disable=True):
            rewards, forecast = experience(
                epsilon, horizon, action_size, agent, dynamics, env
            )
            v = get_discounted_rewards(rewards, agent.gamma)
            df = df.append(
                dict(zip(cols, [epsilon, sum(rewards), v, forecast[horizon:]])),
                ignore_index=True,
            )
            print(epsilon, np.mean([np.mean(x) for x in df[df.epsilon == epsilon].forecast]))
    return df


def main(env_name, n_experiments=1, forecast_horizon=1, epsilons=[0]):
    dynamics = DynamicsModel.load_default_for(env_name)
    agent    = dynamics.agent
    env      = config.get_env(env_name)
    
    action_size = env.action_space.shape[0]
    print("Gamma:", agent.gamma)

    return test_afrl(
        epsilons, forecast_horizon, action_size, agent, dynamics, n_experiments, env
    )


def get_results_folder():
    results_folder = os.path.join(f"data/results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    return results_folder

if __name__ == "__main__":
    env = "LunarLanderContinuous-v2"
    multipliers = np.array([0.001, 0.0025, 0.005, 0.0075, 0.01])
    epsilons = (settings[env]["max_score"] - settings[env]["min_score"]) * multipliers
    horizons = [settings[env]["horizons"][mult] for mult in multipliers]
    df = main(env, 20, horizons, epsilons=epsilons)
    df = df.explode("forecast")
    print(df)
    results_folder = get_results_folder()
    df.to_csv(f"{results_folder}/{env}/experiments.csv")
    # print(df.groupby('epsilon').forecast.mean())
    # print((df.groupby('epsilon').discounted_rewards.mean() - envs[env]['min']) / (envs[env]['max'] - envs[env]['min']))
