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
from info import path_to, config
from train_agent import load_agent
from train_dynamics import load_dynamics
from file_system import FS

def replan(
    state: NDArray,
    old_plan: NDArray,
    forecast: List[int],
    delta: float,
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
            if plan_q + delta < replan_q:
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


def experience(
    delta: float,
    forecast_horizon: int,
    action_size: int,
    agent: OffPolicyAlgorithm,
    dynamics: DynamicsModel,
    env,
):
    episode_forecast = []
    rewards = []
    empty_plan = []
    zero_forecasts = np.zeros(forecast_horizon, np.int8)
    state = env.reset()
    done = False
    plan, forecasts = replan(
        state,
        empty_plan,
        zero_forecasts,
        delta,
        forecast_horizon,
        action_size,
        agent,
        dynamics,
    )
    while not done:
        action = plan[0]
        episode_forecast.append(forecasts[0])
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        plan, forecasts = replan(
            state,
            plan,
            forecasts,
            delta,
            forecast_horizon,
            action_size,
            agent,
            dynamics,
        )
    return rewards, episode_forecast


def get_discounted_rewards(rewards, gamma):
    return sum([r * gamma ** t for t, r in enumerate(rewards)])


def test_afrl(
    deltas: List[float],
    forecast_horizon: int,
    action_size: int,
    agent: OffPolicyAlgorithm,
    dynamics: DynamicsModel,
    n_experiments: int,
    env,
):
    cols = ["delta", "rewards", "discounted_rewards", "forecast"]
    if isinstance(forecast_horizon, list):
        assert len(forecast_horizon) == len(deltas)
    else:
        forecast_horizon = [forecast_horizon] * len(deltas)
    df = pd.DataFrame(columns=cols)
    for delta, horizon in zip(deltas, forecast_horizon):
        for _ in tqdm(range(n_experiments), disable=True):
            rewards, forecast = experience(
                delta, horizon, action_size, agent, dynamics, env
            )
            v = get_discounted_rewards(rewards, agent.gamma)
            df = df.append(
                dict(zip(cols, [delta, sum(rewards), v, forecast[horizon:]])),
                ignore_index=True,
            )
            print(delta, np.mean([np.mean(x) for x in df[df.delta == delta].forecast]))
    return df


def main(env_name, n_experiments=1, forecast_horizon=1, deltas=[0]):
    env = config.get_env(env_name)
    dynamics = load_dynamics(env)

    dynamics.load_state_dict(torch.load(path_to.dynamics_model_for(env_name)))
    action_size = env.action_space.shape[0]
    agent = load_agent(env_name)
    # agent.gamma = 0.95
    print("Gamma:", agent.gamma)

    return test_afrl(
        deltas, forecast_horizon, action_size, agent, dynamics, n_experiments, env
    )


def get_results_folder():
    results_folder = os.path.join(f"data/results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    return results_folder


settings = {
    "LunarLanderContinuous-v2": {
        "max_score": 70,  # (discounted?) ep reward
        "min_score": -100,
        "horizons": {0.001: 3, 0.0025: 5, 0.005: 10, 0.0075: 15, 0.01: 20},
    }
}

if __name__ == "__main__":
    env = "LunarLanderContinuous-v2"
    multipliers = np.array([0.001, 0.0025, 0.005, 0.0075, 0.01])
    deltas = (settings[env]["max_score"] - settings[env]["min_score"]) * multipliers
    horizons = [settings[env]["horizons"][mult] for mult in multipliers]
    df = main(env, 20, horizons, deltas=deltas)
    df = df.explode("forecast")
    print(df)
    results_folder = get_results_folder()
    df.to_csv(f"{results_folder}/{env}/experiments.csv")
    # print(df.groupby('delta').forecast.mean())
    # print((df.groupby('delta').discounted_rewards.mean() - envs[env]['min']) / (envs[env]['max'] - envs[env]['min']))
