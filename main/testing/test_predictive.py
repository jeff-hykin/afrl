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
from training.train_agent import Agent
from training.train_coach import CoachClass
from tools import flatten, get_discounted_rewards, divide_chunks, minibatch, ft, TimestepSeries, to_numpy, average

settings = config.test_predictor

def experience(
    epsilon: float,
    forecast_horizon: int,
    predictor,
):
    episode_forecast = []
    rewards          = []
    empty_plan       = []
    state            = predictor.env.reset()
    forecast         = np.zeros(forecast_horizon+1, np.int8)
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
        for forecast_index, action in enumerate(future_plan):
            # 
            # stopping criteria
            # 
            replan_action = actor_action_for(expected_state)
            replan_q = predictor.agent.value_of(expected_state, replan_action)
            plan_q   = predictor.agent.value_of(expected_state, action)
            # if the planned action is significantly worse, then fail
            if plan_q + epsilon < replan_q:
                break
            
            # 
            # compute next-step 
            # 
            expected_state = predictor.coach.predict(expected_state, action)
            forecast[forecast_index] = forecast[forecast_index + 1] + 1 # for the stats... keep track of the forecast of this action
            new_plan.append(action)

        #
        # for the part that wasnt in the old plan
        #
        for index in range(forecast_index, forecast_horizon):
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
    return rewards, episode_forecast

def main(settings, predictor):
    # 
    # pull in settings
    # 
    multipliers           = list(settings.horizons.keys())
    forecast_horizons     = list(settings.horizons.values())
    sccore_range          = settings.max_score - settings.min_score
    epsilons              = sccore_range * np.array(multipliers)
    predictor.agent.gamma = settings.agent_discount_factor
    
    # define return value
    data = LazyDict(
        epsilon=[],
        rewards=[],
        discounted_rewards=[],
        forecast=[],
        average_forecast=[],
        horizon=[],
    )
    card = ss.DisplayCard("multiLine", dict(
        rewards=[],
        average_forecast=[],
        epsilon=[],
        horizon=[],
    ))
    
    # 
    # perform experiments with all epsilons
    # 
    index = -1
    for epsilon, horizon in zip(epsilons, forecast_horizons):
        for episode_index in range(settings.number_of_episodes):
            rewards, forecast = experience(epsilon, horizon, predictor,)
            average_reward = average(rewards)
            index += 1
            # NOTE: double averaging might not be the desired metric but its probably alright
            # save data
            data.epsilon.append(epsilon)
            data.rewards.append(average_reward)
            data.discounted_rewards.append(get_discounted_rewards(rewards, predictor.agent.gamma))
            data.forecast.append(forecast[horizon:]) # BOOKMARK: I don't understand this part --Jeff
            
            grand_average_forecast = average([
                average(each_forecast)
                    for each_epsilon, each_forecast in zip(data.epsilon, data.forecast)
                        if each_epsilon == epsilon 
            ])
            
            data.average_forecast.append(grand_average_forecast)
            card.send(dict(
                epsilon=[ index, epsilon ],
                average_forecast=[index, grand_average_forecast],
                rewards=[ index, average_reward ],
                horizon=[ index, horizon ],
            ))
            
            print(f"    epsilon: {epsilon:.4f}, average_forecast: {grand_average_forecast:.4f}, average_reward: {average_reward:.2f}")
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