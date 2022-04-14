import os
from types import FunctionType
from typing import List
from copy import deepcopy
from dataclasses import dataclass
import math

import gym
import numpy as np
import pandas as pd
import stable_baselines3 as sb
import torch
import file_system_py as FS
import silver_spectacle as ss
from nptyping import NDArray
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from torch.optim.adam import Adam
from tqdm import tqdm
from trivial_torch_tools import to_tensor, Sequential, init, convert_each_arg
from trivial_torch_tools.generics import to_pure, flatten
from super_map import LazyDict
from simple_namespace import namespace
from rigorous_recorder import RecordKeeper

from info import path_to, config
from tools import flatten, get_discounted_rewards, divide_chunks, minibatch, ft, TimestepSeries, to_numpy, average, median, normalize_rewards, rolling_average
from main.agent import Agent
from main.coach import Coach

settings = config.test_predictor

def experience(
    epsilon: float,
    horizon: int,
    predictor,
):
    episode_forecast = []
    rewards          = []
    failure_points   = []
    state            = predictor.env.reset()
    forecast         = np.zeros(horizon+1, np.int8) # TODO: check if that +1 is wrong
    def replan(
        initial_state: NDArray,
        old_plan: NDArray,
    ):
        nonlocal forecast
        
        actor_action_for = lambda state: predictor.agent.predict(state, deterministic=True)[0]
        
        new_plan = []
        forecast_index = 0
        expected_state = initial_state
        future_plan = old_plan[1:] # reuse old plan (recycle)
        stopped_early = False
        for forecast_index, old_plan_action in enumerate(future_plan):
            # 
            # stopping criteria
            # 
            replan_action = actor_action_for(expected_state)
            replan_q      = predictor.agent.value_of(expected_state, replan_action)
            old_plan_q    = predictor.agent.value_of(expected_state, old_plan_action)
            # if the planned old_plan_action is significantly worse, then fail
            if old_plan_q + epsilon < replan_q:
                stopped_early = True
                break
            
            # 
            # compute next-step 
            # 
            expected_state = predictor.coach.predict(expected_state, old_plan_action)
            forecast[forecast_index] = forecast[forecast_index + 1] + 1 # for the stats... keep track of the forecast of this old_plan_action
            new_plan.append(old_plan_action)

        if stopped_early:
            failure_points.append(forecast_index)
        else:
            failure_points.append(horizon+1)
        
        #
        # for the part that wasnt in the old plan
        #
        for index in range(forecast_index, horizon):
            action = actor_action_for(expected_state)
            expected_state = predictor.coach.predict(expected_state, action)
            new_plan.append(action)
            forecast[index] = 0

        return new_plan, forecast
    
    plan, forecast = replan(state, [])
    done = False
    while not done:
        action = plan[0]
        episode_forecast.append(forecast[0])
        state, reward, done, _ = predictor.env.step(to_numpy(action))
        rewards.append(reward)
        plan, forecast = replan(state, plan,)
    return rewards, episode_forecast, failure_points

def main(settings, predictor):
    # 
    # pull in settings
    # 
    unscaled_epsilon      = list(settings.horizons.keys())
    forecast_horizons     = list(settings.horizons.values())
    reward_range          = settings.max_reward_single_timestep - settings.min_reward_single_timestep
    epsilons              = reward_range * np.array(unscaled_epsilon)
    predictor.agent.gamma = config.train_agent.env_overrides[config.env_name].reward_discount
    print(f'''unscaled_epsilon = {unscaled_epsilon}''')
    print(f'''epsilons = {epsilons}''')
    
    # define return value
    data = LazyDict(
        epsilon=[],
        rewards=[],
        discounted_rewards=[],
        forecast=[],
        alt_forecast=[],
        average_forecast=[],
        alt_average_forecast=[],
        average_failure_point=[],
        horizon=[],
    )
    reward_card = ss.DisplayCard("multiLine", dict(
        rewards=[ (index, each) for index, each in enumerate(data.rewards)               ],
    ))
    prediction_card = ss.DisplayCard("multiLine", dict(
        average_forecast=[],
        alt_average_forecast=[],
        average_failure_point=[],
        epsilon=[],
        horizon=[],
    ))
    
    # 
    # perform experiments with all epsilons
    # 
    reward_timestep_range = settings.max_reward_single_timestep - settings.min_reward_single_timestep
    index = -1
    for epsilon, horizon in zip(epsilons, forecast_horizons):
        for episode_index in range(settings.number_of_episodes):
            rewards, forecast, failure_points = experience(epsilon, horizon, predictor,)
            normalized_rewards = normalize_rewards(rewards, settings.max_reward_single_timestep, settings.min_reward_single_timestep)
            average_failure_point = average(failure_points)
            normalized_episode_reward = sum(normalized_rewards)
            episode_reward = sum(rewards)
            index += 1
            # save data
            data.epsilon.append(epsilon)
            data.rewards.append(normalized_episode_reward)
            data.discounted_rewards.append(get_discounted_rewards(rewards, predictor.agent.gamma))
            data.forecast.append(forecast[horizon:]) # BOOKMARK: I don't understand this part --Jeff
            data.alt_forecast.append(forecast[:horizon])
            data.average_failure_point.append(average_failure_point)
            data.horizon.append(horizon)
            
            # NOTE: double averaging might not be the desired metric but its probably alright
            grand_average_forecast = average([
                average(each_forecast)
                    for each_epsilon, each_forecast in zip(data.epsilon, data.forecast)
                        if each_epsilon == epsilon 
            ])
            alt_average_forecast = average([
                average(each_forecast)
                    for each_epsilon, each_forecast in zip(data.epsilon, data.alt_forecast)
                        if each_epsilon == epsilon 
            ])
            
            data.average_forecast.append(grand_average_forecast)
            data.alt_average_forecast.append(alt_average_forecast)
            reward_card.send(dict(
                rewards=[ index, episode_reward ],
            ))
            prediction_card.send(dict(
                epsilon=[ index, epsilon ],
                average_forecast=[index, grand_average_forecast],
                alt_average_forecast=[index, alt_average_forecast],
                average_failure_point=[index, average_failure_point],
                horizon=[ index, horizon ],
            ))
            
            print(f"    epsilon: {epsilon:.4f}, average_forecast: {grand_average_forecast:.4f}, episode_reward:{sum(rewards):.2f}, normalized_episode_reward: {normalized_episode_reward:.4f}, max_timestep_reward: {max(rewards):.2f}, min_timestep_reward: {min(rewards):.2f}")
    smoothing = settings.graph_smoothing
    # display cards at the end with the final data (the other card is transient)
    ss.DisplayCard("multiLine", dict(
        rewards=list(enumerate(rolling_average(data.rewards, smoothing))),
    ))
    ss.DisplayCard("multiLine", dict(
        average_forecast=      list(enumerate(rolling_average(data.average_forecast      , smoothing))),
        alt_average_forecast=  list(enumerate(rolling_average(data.alt_average_forecast  , smoothing))),
        average_failure_point= list(enumerate(rolling_average(data.average_failure_point , smoothing))),
        epsilon=               list(enumerate(rolling_average(data.epsilon               , smoothing))),
        horizon=               list(enumerate(rolling_average(data.horizon               , smoothing))),
    ))
    return data

def run_test(env_name, coach, csv_path):
    print(f'''\n\n-----------------------------------------------------------------------------------------------------''')
    print(f''' Testing Agent+Coach''')
    print(f'''-----------------------------------------------------------------------------------------------------\n\n''')
    # compute data
    data = main(
        settings=settings.merge(settings.env_overrides[env_name]),
        predictor=LazyDict(
            env=config.get_env(env_name),
            coach=coach,
            agent=coach.agent,
        ),
    )
    
    # export to CSV
    FS.clear_a_path_for(csv_path, overwrite=True)
    pd.DataFrame(data).explode("forecast").to_csv(csv_path)
    return data