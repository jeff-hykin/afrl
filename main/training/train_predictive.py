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
from torch.optim.adam import Adam
from tqdm import tqdm
import file_system_py as FS
from trivial_torch_tools import to_tensor, Sequential, init, convert_each_arg
from trivial_torch_tools.generics import to_pure
from super_map import LazyDict

from info import path_to, config
from main.training.train_agent import Agent
from main.training.train_dynamics import DynamicsModel
from main.tools import flatten, get_discounted_rewards, divide_chunks, minibatch, ft, TimestepSeries

settings = config.gym_env_settings

class PredictiveTest():

    def __init__(self, env_name):
        self.timesteps = TimestepSeries()
        self.horizon = config.train_predictive.initial_horizon_size
        self.loss_threshold = config.train_predictive.loss_threshold
        self.record = LazyDict(
            losses=[],
            horizons=[],
        )
        # 
        # load models
        # 
        self.dynamics = DynamicsModel.load_default_for(env_name)
        self.dynamics.which_loss = "timestep_loss"
        self.agent    = dynamics.agent
        self.env = config.get_env(env_name)
    
    def run(self, env_name, number_of_epochs):
        for epoch_index in range(number_of_epochs):
            self.timesteps = TimestepSeries()
            next_state = self.env.reset()
            done = False
            while not done:
                state = next_state
                action = self.agent.make_decision(state)
                next_state, reward, done, _ = self.env.step(to_pure(action))
                self.timesteps.add(state, action, reward, next_state)
                self.check_forcast()
    
    def check_forcast(self):
        # "what would have been predicted X horizon ago"
        if len(self.timesteps.steps) > self.horizon:
            time_slice = self.timesteps[-self.horizon:]
            loss = self.dynamics.training_loss(time_slice)
            self.record.losses.append(to_pure(loss))
            self.record.horizons.append(self.horizon)
            self.update_horizon()
    
    def update_horizon(self):
        if self.record.losses[-1] < self.loss_threshold:
            self.horizon -= 1
        else:
            self.horizon += 1

def replan(
    state: NDArray,
    old_plan: NDArray,
    forecast: List[int],
    epsilon: float,
    forecast_horizon: int,
    action_size: int,
    agent: OffPolicyAlgorithm,
    dynamics: DynamicsModel,
    predpolicy,
    optimizer,
):
    global losses

    new_plan = []
    k = 0  # used to keep track of forecast of the actions

    # reuse old plan (recycle)
    for (pred_state, action) in old_plan[1:]:
        replan_action = agent.predict(state, deterministic=True)[0]
        with torch.no_grad():
            replan_q = agent.value_of(state, replan_action)
        plan_action = predpolicy(ft([pred_state]))[0]
        plan_q = agent.value_of(state, plan_action)
        diff = replan_q - plan_q
        losses.append(diff)
        if len(losses) == 32: # QUESTION: what is this 32? minibatch_size? or is the statement checking if training-mode vs testing-mode?
            optimizer.zero_grad()
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            losses = []

        if diff.item() > epsilon:
            break
        new_plan.append((pred_state, plan_action))
        state = dynamics(state, plan_action)
        # for the stats... keep track of the forecast of this action
        forecast[k] = forecast[k + 1] + 1
        k += 1

    # produce new plan (replan)
    for i in range(k, forecast_horizon):
        action = predpolicy(ft([state]))[0]
        new_plan.append((state, action))
        with torch.no_grad():
            state = dynamics(state, action)
        forecast[i] = 0

    return new_plan, forecast


def experience(
    epsilon: float,
    forecast_horizon: int,
    action_size: int,
    agent: OffPolicyAlgorithm,
    predpolicy,
    dynamics: DynamicsModel,
    env,
    optimizer,
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
        epsilon,
        forecast_horizon,
        action_size,
        agent,
        predpolicy,
        dynamics,
        optimizer,
    )
    while not done:
        action = plan[0][1]
        episode_forecast.append(forecasts[0])
        state, reward, done, _ = env.step(action.detach().to(torch.device('cpu')).numpy())
        rewards.append(reward)
        plan, forecasts = replan(
            state,
            plan,
            forecasts,
            epsilon,
            forecast_horizon,
            action_size,
            agent,
            predpolicy,
            dynamics,
            optimizer,
        )
    return rewards, episode_forecast


def test_afrl(
    epsilons: List[float],
    forecast_horizon: int,
    action_size: int,
    agent: OffPolicyAlgorithm,
    predpolicy,
    dynamics: DynamicsModel,
    n_experiments: int,
    env,
    optimizer,
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
                epsilon,
                horizon,
                action_size,
                agent,
                predpolicy,
                dynamics,
                env,
                optimizer,
            )
            v = get_discounted_rewards(rewards, agent.gamma)
            df = df.append(
                dict(zip(cols, [epsilon, sum(rewards), v, forecast[horizon:]])),
                ignore_index=True,
            )
            print(
                epsilon,
                np.mean([np.mean(x) for x in df[df.epsilon == epsilon].forecast]),
            )
    return df


def main(env_name, n_experiments=1, forecast_horizon=1, epsilons=[0]):
    dynamics = DynamicsModel.load_default_for(env_name)
    agent    = dynamics.agent
    env = config.get_env(env_name)
    
    action_size = env.action_space.shape[0]
    predpolicy  = deepcopy(agent.policy)
    optimizer   = Adam(predpolicy.parameters(), lr=0.0001)

    # predpolicy.load_state_dict(torch.load(f'data/models/agents/predpolicy/{env_name}.pt'))

    print("Gamma:", agent.gamma)

    return test_afrl(
        epsilons,
        forecast_horizon,
        action_size,
        agent,
        predpolicy,
        dynamics,
        n_experiments,
        env,
        optimizer,
    )

if __name__ == "__main__":
    from copy import deepcopy
    
    for env_name in config.env_names:
        multipliers = np.array(list(settings[env_name]["horizons"].keys()))
        epsilons = (settings[env_name]["max_score"] - settings[env_name]["min_score"]) * multipliers
        horizons = [settings[env_name]["horizons"][mult] for mult in multipliers]
        
        df = main(
            env_name,
            config.train_predictive.number_of_experiments,
            horizons,
            epsilons=epsilons,
        ).explode("forecast")
        df.to_csv(
            FS.ensure_folder_exists(
                FS.parent_folder(
                    f"{path_to.folder.results}/{env_name}/experiments.csv"
                )
            )
        )
        # print(df.groupby('epsilon').forecast.mean())
        # print((df.groupby('epsilon').discounted_rewards.mean() - envs[env]['min']) / (envs[env]['max'] - envs[env]['min']))
