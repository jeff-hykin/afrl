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

# 
# combine Agent,Dynamics,Env so they dont get mis-matched
# 
from dataclasses import dataclass
@dataclass
class PredictorEnv:
    env_name             = None
    env                  = None
    dynamics             = None
    agent                = None
    actor_copy           = None
    actor_copy_optimizer = None
    
    # default values:
    def __init__(self, **kwargs):
        for name, value in kwargs.items(): setattr(self, name, value)
        if self.env_name:
            if not self.env                  : self.env                  = config.get_env(self.env_name)
            if not self.dynamics             : self.dynamics             = DynamicsModel.load_default_for(self.env_name)
            if not self.agent                : self.agent                = self.dynamics.agent
            if not self.actor_copy           : self.actor_copy           = deepcopy(self.agent.policy)
            if not self.actor_copy_optimizer : self.actor_copy_optimizer = Adam(self.actor_copy.parameters(), lr=config.gym_env_settings[self.env_name].actor_copy_learning_rate) 

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
        self.dynamics            = DynamicsModel.load_default_for(env_name)
        self.dynamics.which_loss = "timestep_loss"
        self.agent               = self.dynamics.agent
        self.env                 = config.get_env(env_name)
    
    def run(self, number_of_epochs):
        for epoch_index in range(number_of_epochs):
            self.timesteps = TimestepSeries()
            next_state = self.env.reset()
            done = False
            while not done:
                state = next_state
                action = self.agent.make_decision(state)
                next_state, reward, done, _ = self.env.step(to_pure(action))
                self.timesteps.add(state, action, reward, next_state)
                self.check_forecast()
    
    def check_forecast(self):
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

def experience(
    epsilon: float,
    forecast_horizon: int,
    predictor: PredictorEnv,
):
    episode_forecast = []
    rewards          = []
    loss_batch       = []
    state            = predictor.env.reset()
    forecast         = np.zeros(forecast_horizon+1, np.int8)
    
    # main helper
    def replan(intial_state: NDArray, old_plan: NDArray,):
        nonlocal forecast
        # 
        # inner helpers
        # 
        def get_actor_action_for(state):
            action, _ = predictor.agent.predict(state, deterministic=True)
            return action 
        
        def get_actor_copy_action_for(state):
            action = predictor.actor_copy(to_tensor([state]))[0]
            return action 
        
        def without_grad_get_value(state, action):
            with torch.no_grad():
                return predictor.agent.value_of(state, action)
        
        def with_grad_get_value(state, action):
            return predictor.agent.value_of(state, action)
        
        def get_predicted_next_state(state, action):
            return predictor.dynamics.predict(state, action)
        
        def run_backprop_if_ready():
            if len(loss_batch) >= config.train_predictive.weight_update_frequency:
                predictor.actor_copy_optimizer.zero_grad()
                loss = torch.stack(loss_batch).mean()
                loss.backward()
                predictor.actor_copy_optimizer.step()
                loss_batch.clear()
        
        # 
        # replan core
        # 
        new_plan = []

        state = intial_state
        future_plan = old_plan[1:] # reuse old plan (recycle)
        forecast_index = 0
        for forecast_index, (predicted_state, planned_action) in enumerate(future_plan):
            # 
            # inline loss
            # 
            replan_action = get_actor_action_for(     state)
            plan_action   = get_actor_copy_action_for(state) # QUESTION: why actor copy (predpolicy) and not actor (agent)?
            value_of_plan_action   = with_grad_get_value(   state, plan_action  )
            value_of_replan_action = without_grad_get_value(state, replan_action) # QUESTION: why only this one without grad
            loss = how_much_better_was_replanned_action = value_of_replan_action - value_of_plan_action
            
            # backprop
            loss_batch.append(loss)
            run_backprop_if_ready()
            
            # exit condition
            if to_pure(loss) > epsilon:
                break
            
            new_plan.append((predicted_state, plan_action))
            # BOOKMARK: check this (the no_grad in the function call)
            # #seems like it could be problematic (expanded from DynamicsModel.forward)
            state = get_predicted_next_state(state, plan_action) # note use of plan_action
            # the likely reason this no_grad is here is 
            # so that future states can be predicted 
            # without penalizing the first one for +2 timestep-loss, +3 timestep loss, +4 timestep loss, etc
            # just generate the state, then redo +1 timestep-loss only
            # except I cant find anywhere with grad for the dynamics model
            forecast[forecast_index] = forecast[forecast_index+1] + 1 # for the stats... keep track of the forecast of this action
        
        #
        # for the part that wasnt in the old plan
        #
        for index in range(forecast_index, forecast_horizon):
            action = get_actor_copy_action_for(state)
            new_plan.append((state, action))
            
            state = get_predicted_next_state(state, action)
            # # Below is what used to be ^
            # with torch.no_grad(): # this is a double-no-grad cause there's already a no-grad inside dynamics.forward()
            #     state = dynamics(state, action)
            forecast[index] = 0

        return new_plan, forecast
    
    # 
    # runtime
    # 
    plan, forecast = replan(state, [])
    done = False
    while not done:
        predicted_state, action = plan[0]
        episode_forecast.append(forecast[0])
        state, reward, done, _ = predictor.env.step(to_numpy(action))
        rewards.append(reward)
        plan, forecast = replan(state, plan,)
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

if __name__ == "__main__":
    for env_name in config.env_names:
        # compute data
        data = main(
            settings=config.gym_env_settings[env_name],
            predictor=PredictorEnv(env_name=env_name),
        )
        
        # export to CSV
        csv_path = path_to.experiment_csv_for(env_name)
        FS.ensure_is_folder(FS.parent_folder(csv_path))
        pd.DataFrame(data).explode("forecast").to_csv(csv_path)