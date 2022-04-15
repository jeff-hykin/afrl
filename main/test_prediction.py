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
from trivial_torch_tools.generics import to_pure, flatten, large_pickle_load, large_pickle_save
from super_map import LazyDict
from simple_namespace import namespace
from rigorous_recorder import Recorder

from info import path_to, config
from tools import get_discounted_rewards, divide_chunks, minibatch, ft, TimestepSeries, to_numpy, average, median, normalize, rolling_average, key_prepend, simple_stats
from main.agent import Agent
from main.coach import Coach

class Tester:
    def __init__(self, settings, predictor, path, csv_path, attribute_overrides={}):
        self.settings = settings
        self.predictor = predictor
        self.path = path
        self.csv_path = csv_path
        self.csv_data = None
        self.reward_card = None
        self.prediction_card = None
        # Data recording below is independent to reduce max file size (lets it sync to git)
        self.recorder = Recorder()
        self.rewards_per_episode_per_timestep            = [None] * settings.number_of_episodes
        self.discounted_rewards_per_episode_per_timestep = [None] * settings.number_of_episodes
        self.failure_points_per_episode_per_timestep     = [None] * settings.number_of_episodes
        self.stopped_early_per_episode_per_timestep      = [None] * settings.number_of_episodes
        self.real_q_values_per_episode_per_timestep      = [None] * settings.number_of_episodes
        
        # for loading from a file
        for each_key, each_value in attribute_overrides.items():
            setattr(self, each_key, each_value)
    
    # 
    # core algorithm
    # 
    def experience_episode(
        self,
        epsilon: float,
        horizon: int,
        episode_index: int,
    ):
        # data recording
        rewards_per_timestep            = []
        discounted_rewards_per_timestep = []
        failure_points_per_timestep     = []
        stopped_early_per_timestep      = []
        real_q_values_per_timestep      = []
        
        # setup shortcuts/naming
        env                = self.predictor.env
        predict_next_state = self.predictor.coach.predict
        actor_model        = self.predictor.agent.predict
        value_batch        = self.predictor.agent.value_of
        choose_action      = lambda state: actor_model(state, deterministic=True)[0]
        q_value_for        = lambda state, action: value_batch(state, action)[0][0]
        reward_discount    = self.predictor.agent.gamma
        
        # inline def so it has access to shortcuts above
        def replan(initial_state, old_plan):
            new_plan = []
            forecast_index = 0
            expected_state = initial_state
            future_plan = old_plan[1:] # reuse old plan (recycle)
            stopped_early = False
            for forecast_index, old_plan_action in enumerate(future_plan):
                # 
                # stopping criteria
                # 
                replan_action = choose_action(expected_state)
                replan_q      = q_value_for(expected_state, replan_action)
                old_plan_q    = q_value_for(expected_state, old_plan_action)
                # if the planned old_plan_action is significantly worse, then fail
                if old_plan_q + epsilon < replan_q:
                    stopped_early = True
                    break
                
                # 
                # compute next-step 
                # 
                expected_state = predict_next_state(expected_state, old_plan_action)
                new_plan.append(old_plan_action)
            
            failure_point = forecast_index
            
            #
            # for the part that wasnt in the old plan
            #
            for index in range(forecast_index, horizon):
                action = choose_action(expected_state)
                expected_state = predict_next_state(expected_state, action)
                new_plan.append(action)

            return new_plan, failure_point, stopped_early
        
        # init values
        done     = False
        timestep = -1
        state    = env.reset()
        episode_forecast = []
        rolling_forecast = [0]*(horizon+1) # TODO: check if that +1 is wrong (if right, removing should error)
        rolling_forecast = self.update_rolling_forecast(rolling_forecast, horizon)
        plan, failure_point, stopped_early = replan(state, [])
        while not done:
            timestep += 1
            
            action = plan[0]
            real_q_value = q_value_for(state, action)
            episode_forecast.append(rolling_forecast[0])
            
            state, reward, done, _ = env.step(to_numpy(action))
            plan, failure_point, stopped_early = replan(state, plan,)
            rolling_forecast = self.update_rolling_forecast(rolling_forecast, failure_point)
            
            # record data
            rewards_per_timestep.append(to_pure(reward))
            discounted_rewards_per_timestep.append(to_pure(reward * (reward_discount ** timestep)))
            failure_points_per_timestep.append(to_pure(failure_point))
            stopped_early_per_timestep.append(to_pure(stopped_early))
            real_q_values_per_timestep.append(to_pure(real_q_value))
            
        # data recording (and convert to tuple to reduce memory pressure)
        rewards_per_timestep            = self.rewards_per_episode_per_timestep[episode_index]            = tuple(rewards_per_timestep)
        discounted_rewards_per_timestep = self.discounted_rewards_per_episode_per_timestep[episode_index] = tuple(discounted_rewards_per_timestep)
        failure_points_per_timestep     = self.failure_points_per_episode_per_timestep[episode_index]     = tuple(failure_points_per_timestep)
        stopped_early_per_timestep      = self.stopped_early_per_episode_per_timestep[episode_index]      = tuple(stopped_early_per_timestep)
        real_q_values_per_timestep      = self.real_q_values_per_episode_per_timestep[episode_index]      = tuple(real_q_values_per_timestep)
        return episode_forecast, rewards_per_timestep, discounted_rewards_per_timestep, failure_points_per_timestep, stopped_early_per_timestep, real_q_values_per_timestep

    # 
    # setup for testing
    # 
    def run_all_episodes(self):
        print(f'''\n\n-----------------------------------------------------------------------------------------------------''')
        print(f''' Testing Agent+Coach''')
        print(f'''-----------------------------------------------------------------------------------------------------\n\n''')
        settings, predictor = self.settings, self.predictor
        # 
        # pull in settings
        # 
        normal_epsilons     = tuple(settings.horizons.keys())
        forecast_horizons     = tuple(settings.horizons.values())
        reward_range          = settings.max_reward_single_timestep - settings.min_reward_single_timestep
        epsilons              = reward_range * np.array(normal_epsilons)
        predictor.agent.gamma = config.train_agent.env_overrides[config.env_name].reward_discount
        print(f'''normal_epsilons = {normal_epsilons}''')
        print(f'''epsilons = {epsilons}''')
        
        # define return value
        self.csv_data = LazyDict(
            epsilon=[],
            rewards=[],
            discounted_rewards=[],
            forecast=[],
        )
        
        # 
        # perform experiments with all epsilons
        # 
        index = -1
        for epsilon, horizon, normal_epsilon in zip(epsilons, forecast_horizons, normal_epsilons):
            
            epoch_recorder = Recorder(
                horizon=horizon,
                scaled_epsilon=epsilon,
                normal_epsilon=normal_epsilon,
            ).set_parent(self.recorder)
            
            forecast_slices = []
            alt_forecast_slices = []
            
            for episode_index in range(settings.number_of_episodes):
                index += 1
                forecast, rewards, discounted_rewards, failure_points, stopped_earlies, real_q_values = self.experience_episode(epsilon, horizon, episode_index)
                
                reward_stats            = simple_stats(rewards)
                discounted_reward_stats = simple_stats(discounted_rewards)
                failure_point_stats     = simple_stats(failure_points)
                real_q_value_stats      = simple_stats(real_q_values)
                
                normalized_rewards = normalize(rewards, min=settings.min_reward_single_timestep, max=settings.max_reward_single_timestep)
                
                # 
                # forecast stats
                # 
                forecast_slices.append(forecast[horizon:])
                alt_forecast_slices.append(forecast)
                # NOTE: double averaging might not be the desired metric but its probably alright
                grand_forecast_average = average(average(each) for each in forecast_slices)
                alt_forecast_average   = average(average(each) for each in alt_forecast_slices)
                
                # save self.csv_data
                epoch_recorder.push(
                    episode_index=episode_index,
                    timestep_count=len(rewards),
                    forecast_average=grand_forecast_average,
                    alt_forecast_average=alt_forecast_average,
                    normalized_reward_average=simple_stats(normalized_rewards).average,
                    **key_prepend("reward", reward_stats), # reward_average, reward_median, reward_min, reward_max, etc
                    **key_prepend("discounted_reward", discounted_reward_stats), # discounted_reward_average, discounted_reward_median, discounted_reward_min, discounted_reward_max, etc
                    **key_prepend("failure_point", failure_point_stats),
                    **key_prepend("q", real_q_value_stats),
                )
                self.increment_live_graphs() # uses latest record
                
                # record for CSV backwards compatibility
                self.csv_data.epsilon.append(epsilon)
                self.csv_data.rewards.append(reward_stats.sum)
                self.csv_data.discounted_rewards.append(discounted_reward_stats.sum)
                self.csv_data.forecast.append(forecast[horizon:]) # BOOKMARK: len(forecast) == number_of_timesteps, so I have no idea why horizon is being used to slice it
                
                print(f"    epsilon: {epsilon:.4f}, forecast_average: {grand_forecast_average:.4f}, episode_reward:{reward_stats.sum:.2f}, max_timestep_reward: {reward_stats.max:.2f}, min_timestep_reward: {reward_stats.min:.2f}")
        
        # display cards at the end with the final self.csv_data (the other card is transient)
        self.generate_graphs() # uses self.recorder
        
        return self

    # 
    # misc helpers
    # 
    def generate_graphs(self):
        smoothing = self.settings.graph_smoothing
        
        timestep_reward_averages     = []
        q_values               = []
        scaled_epsilons        = []
        forecasts_average      = []
        failure_points_average = []
        horizons               = []
        
        # for each epsilon-horizon pair
        for each_recorder in self.recorder.sub_recorders:
            timestep_reward_averages     += rolling_average(each_recorder.frame["reward_average"]           , smoothing)
            q_values               += rolling_average(each_recorder.frame["q_average"]                , smoothing)
            forecasts_average      += rolling_average(each_recorder.frame["forecast_average"]         , smoothing)
            failure_points_average += rolling_average(each_recorder.frame["failure_point_average"]    , smoothing)
            scaled_epsilons        += [ each_recorder["scaled_epsilon"] ]*len(timestep_reward_averages)
            horizons               += [ each_recorder["horizon"]        ]*len(timestep_reward_averages)
        
        # add indicies to all of them
        timestep_reward_averages     = tuple(enumerate(timestep_reward_averages    ))
        q_values               = tuple(enumerate(q_values              ))
        scaled_epsilons        = tuple(enumerate(scaled_epsilons       ))
        forecasts_average      = tuple(enumerate(forecasts_average     ))
        failure_points_average = tuple(enumerate(failure_points_average))
        horizons               = tuple(enumerate(horizons              ))
        
        # 
        # display the actual cards
        # 
        reward_card = ss.DisplayCard("multiLine", dict(
            scaled_epsilon=scaled_epsilons,
            timestep_reward_average=timestep_reward_averages,
            timestep_q_average=q_values,
        ))
        prediction_card = ss.DisplayCard("multiLine", dict(
            forecast_average=forecasts_average,
            failure_point_average=failure_points_average,
            horizon=horizons,
        ))
        text_card = ss.DisplayCard("quickMarkdown", f"""## {config.experiment_name}""")
    
    def init_live_graphs(self):
        self.reward_card = ss.DisplayCard("multiLine", dict(
            scaled_epsilon=[],
            timestep_reward_average=[],
            timestep_q_average=[],
        ))
        self.prediction_card = ss.DisplayCard("multiLine", dict(
            forecast_average=[],
            failure_point_average=[],
            horizon=[],
        ))
        self.text_card = ss.DisplayCard("quickMarkdown", f"""## {config.experiment_name}""")
    
    def increment_live_graphs(self):
        if not self.reward_card: self.init_live_graphs()
        
        index = sum(len(each) for each in self.recorder.sub_recorders)
        latest_record = self.recorder.sub_recorders[-1][-1]
        
        self.reward_card.send(dict(
            timestep_reward_average=[index, latest_record["reward_average"] ],
            timestep_q_average=[index, latest_record["q_average"] ],
            scaled_epsilon=[index, latest_record["scaled_epsilon"] ],
        ))
        self.prediction_card.send(dict(
            forecast_average=[index, latest_record["forecast_average"] ],
            failure_point_average=[index, latest_record["failure_point_average"] ],
            horizon=[index, latest_record["horizon"] ],
        ))
    
    def update_rolling_forecast(self, old_rolling_forecast, failure_point):
        pairwise = zip(old_rolling_forecast[0:-1], old_rolling_forecast[1:])
        
        # each = next+1
        new_rolling_forecast = [ next+1 for current, next in pairwise ]
        new_rolling_forecast.append(0) # pairwise is always 1 element shorter than original, so add missing element
        # zero-out everything past the failure point
        for index in range(failure_point, len(new_rolling_forecast)):
            new_rolling_forecast[index] = 0
            
        return new_rolling_forecast
    
    # 
    # save and load methods
    # 
    attributes_to_save = [
        "settings",
        "recorder",
        "rewards_per_episode_per_timestep",
        "discounted_rewards_per_episode_per_timestep",
        "failure_points_per_episode_per_timestep",
        "stopped_early_per_episode_per_timestep",
        "real_q_values_per_episode_per_timestep",
    ]
    
    @classmethod
    def load(cls, path):
        attributes = {}
        for each_attribute_name in cls.attributes_to_save:
            attributes[each_attribute_name] = large_pickle_load(f"{path}/{each_attribute_name}.pickle")
        # create a tester with the loaded data
        return Tester(
            settings=attributes["settings"],
            predictor=None,
            attribute_overrides=attributes,
        )
    
    def save(self, path=None):
        path = path or self.path
        # save normal things
        for each_attribute_name in self.attributes_to_save:
            each_path = f"{path}/{each_attribute_name}.pickle"
            FS.clear_a_path_for(each_path, overwrite=True)
            large_pickle_save(getattr(self, each_attribute_name, None), each_path)
        # save csv
        FS.clear_a_path_for(self.csv_path, overwrite=True)
        pd.DataFrame(self.csv_data).explode("forecast").to_csv(csv_path)
        return self
            