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
from rigorous_recorder import RecordKeeper

from info import path_to, config
from training.train_agent import Agent
from training.train_coach import CoachClass
from tools import flatten, get_discounted_rewards, divide_chunks, minibatch, ft, TimestepSeries, to_numpy, average

# 
# combine Agent,Coach,Env so they dont get mis-matched
# 
from dataclasses import dataclass
@dataclass
class PredictorEnv:
    env_name = None
    env      = None
    coach    = None
    agent    = None


def experience(
    epsilon: float,
    forecast_horizon: int,
    predictor: PredictorEnv,
):
    episode_forecast = []
    rewards          = []
    empty_plan       = []
    state            = predictor.env.reset()
    forecast         = np.zeros(forecast_horizon+1, np.int8)
    def replan(
        intial_state: NDArray,
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
            new_plan[forecast_index] = action

        #
        # for the part that wasnt in the old plan
        #
        for index in range(forecast_index, forecast_horizon):
            action = actor_action_for(expected_state)
            expected_state = predictor.coach.predict(expected_state, action)
            new_plan[index] = action
            forecast[index] = 0

        return new_plan, forecast
    
    plan, forecast = replan(state, [])
    done = False
    while not done:
        action = plan[0]
        episode_forecast.append(forecasts[0])
        state, reward, done, _ = predictor.env.step(to_numpy(action))
        rewards.append(reward)
        plan, forecasts = replan(state, plan,)
    return rewards, episode_forecast

def main(
    settings: LazyDict,
    predictor: PredictorEnv,
):
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
    )
    # 
    # perform experiments with all epsilons
    # 
    for epsilon, horizon in zip(epsilons, forecast_horizons):
        for experiment_index in range(settings.number_of_experiments):
            rewards, forecast = experience(epsilon, horizon, predictor,)
            # save data
            data.epsilon.append(epsilon)
            data.rewards.append(sum(rewards))
            data.discounted_rewards.append(get_discounted_rewards(rewards, predictor.agent.gamma))
            data.forecast.append(forecast[horizon:])
            
            # NOTE: double averaging might not be the desired metric but alright
            grand_average_forecast = average([
                average(each_forecast)
                    for each_epsilon, each_forecast in zip(data.epsilon, data.forecast)
                        if each_epsilon == epsilon 
            ])
            print(epsilon, grand_average_forecast)
    return data

def run_test(env_name, coach, csv_path):
    # compute data
    data = main(
        settings=config.gym_env_settings[env_name],
        predictor=PredictorEnv(
            env_name=env_name,
            coach=coach,
            agent=coach.agent,
        ),
    )
    
    # export to CSV
    FS.clear_a_path_for(csv_path, overwrite=True)
    pd.DataFrame(data).explode("forecast").to_csv(csv_path)
    return data