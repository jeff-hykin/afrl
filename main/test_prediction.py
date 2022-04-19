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
from tools import get_discounted_rewards, divide_chunks, minibatch, ft, TimestepSeries, to_numpy, average, median, normalize, rolling_average, key_prepend, simple_stats, log_scale
from smart_cache import cache
from main.agent import Agent
from main.coach import Coach

class Tester:
    def __init__(self, settings, predictor, path, attribute_overrides={}):
        self.settings = settings
        self.predictor = predictor
        self.path = path
        self.csv_path = f"{path}/experiments.csv"
        self.agent_reward_discount = self.predictor.agent.gamma if self.predictor else None
        self.csv_data = None
        self.threshold_card = None
        self.prediction_card = None
        # Data recording below is independent to reduce max file size (lets it sync to git)
        self.recorder = Recorder()
        self.rewards_per_episode_per_timestep            = [None] * settings.number_of_episodes_for_testing
        self.discounted_rewards_per_episode_per_timestep = [None] * settings.number_of_episodes_for_testing
        self.failure_points_per_episode_per_timestep     = [None] * settings.number_of_episodes_for_testing
        self.stopped_early_per_episode_per_timestep      = [None] * settings.number_of_episodes_for_testing
        self.real_q_values_per_episode_per_timestep      = [None] * settings.number_of_episodes_for_testing
        self.q_value_gaps_per_episode_per_timestep       = [None] * settings.number_of_episodes_for_testing
        
        # for loading from a file
        for each_key, each_value in attribute_overrides.items():
            setattr(self, each_key, each_value)
        
        # for saving to yaml
        self.simple_data = LazyDict(
            number_of_episodes_for_testing=self.settings.number_of_episodes_for_testing,
            acceptable_performance_loss=self.settings.acceptable_performance_loss,
            inital_epsilon=self.settings.inital_epsilon,
            intial_horizon=self.settings.intial_horizon,
        )
    
    # 
    # core algorithm
    # 
    def experience_episode(
        self,
        scaled_epsilon: float,
        horizon: int,
        episode_index: int,
    ):
        # data recording
        rewards_per_timestep            = []
        discounted_rewards_per_timestep = []
        failure_points_per_timestep     = []
        stopped_early_per_timestep      = []
        real_q_values_per_timestep      = []
        q_value_gaps_per_timestep       = []
        
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
            q_value_gaps = []
            for forecast_index, old_plan_action in enumerate(future_plan):
                # 
                # stopping criteria
                # 
                replan_action = choose_action(expected_state)
                replan_q      = q_value_for(expected_state, replan_action)
                old_plan_q    = q_value_for(expected_state, old_plan_action)
                q_value_gap = to_pure(replan_q - old_plan_q)
                q_value_gaps.append(q_value_gap)
                if q_value_gap is None:
                    raise Exception(f'''
                        replan_q: {replan_q}
                        old_plan_q: {old_plan_q}
                        replan_q - old_plan_q: {replan_q - old_plan_q}
                        to_pure(replan_q - old_plan_q): {to_pure(replan_q - old_plan_q)}
                        q_value_gaps: {q_value_gaps}
                    ''')
                # if the planned old_plan_action is significantly worse, then fail
                if q_value_gap > scaled_epsilon:
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

            return new_plan, failure_point, stopped_early, tuple(q_value_gaps)
        
        # init values
        done     = False
        timestep = -1
        state    = env.reset()
        episode_forecast = []
        rolling_forecast = [0]*(horizon+1) # TODO: check if that +1 is wrong (if right, removing should error)
        rolling_forecast = self.update_rolling_forecast(rolling_forecast, horizon)
        plan, failure_point, stopped_early, q_value_gaps = replan(state, [])
        while not done:
            timestep += 1
            
            action = plan[0]
            real_q_value = q_value_for(state, action)
            episode_forecast.append(rolling_forecast[0])
            
            state, reward, done, _ = env.step(to_numpy(action))
            plan, failure_point, stopped_early, q_value_gaps = replan(state, plan,)
            rolling_forecast = self.update_rolling_forecast(rolling_forecast, failure_point)
            
            # record data
            rewards_per_timestep.append(to_pure(reward))
            discounted_rewards_per_timestep.append(to_pure(reward * (reward_discount ** timestep)))
            failure_points_per_timestep.append(to_pure(failure_point))
            stopped_early_per_timestep.append(to_pure(stopped_early))
            real_q_values_per_timestep.append(to_pure(real_q_value))
            q_value_gaps_per_timestep.append(q_value_gaps)
            
        # data recording (and convert to tuple to reduce memory pressure)
        rewards_per_timestep            = self.rewards_per_episode_per_timestep[episode_index]            = tuple(rewards_per_timestep)
        discounted_rewards_per_timestep = self.discounted_rewards_per_episode_per_timestep[episode_index] = tuple(discounted_rewards_per_timestep)
        failure_points_per_timestep     = self.failure_points_per_episode_per_timestep[episode_index]     = tuple(failure_points_per_timestep)
        stopped_early_per_timestep      = self.stopped_early_per_episode_per_timestep[episode_index]      = tuple(stopped_early_per_timestep)
        real_q_values_per_timestep      = self.real_q_values_per_episode_per_timestep[episode_index]      = tuple(real_q_values_per_timestep)
        q_value_gaps_per_timestep       = self.q_value_gaps_per_episode_per_timestep[episode_index]       = tuple(q_value_gaps_per_timestep)
        return episode_forecast, rewards_per_timestep, discounted_rewards_per_timestep, failure_points_per_timestep, stopped_early_per_timestep, real_q_values_per_timestep, q_value_gaps_per_timestep
    
    # decides when cache-busting happends (happens if any of these change)
    def __super_hash__(self):
        return (
            self.path,
            self.agent_reward_discount,
            self.settings.number_of_episodes_for_testing,
            self.settings.acceptable_performance_loss,
            self.settings.inital_epsilon,
            self.settings.intial_horizon,
            self.settings.number_of_episodes_for_baseline,
            self.settings.number_of_episodes_for_optimal_parameters,
        )
    
    @cache()
    def gather_baseline(self):
        print("----- getting a reward baseline -----------------------------------------------------------------------------------------------------------------------------------")
        discounted_rewards_per_episode = []
        for episode_index in range(self.settings.number_of_episodes_for_baseline):
            _, rewards, discounted_rewards, _, _, _, q_value_gaps = self.experience_episode(scaled_epsilon=0, horizon=1, episode_index=episode_index)
            total = sum(discounted_rewards)
            discounted_rewards_per_episode.append(total)
            print(f'''  episode_index={episode_index}, total={total}''')
        baseline = simple_stats(discounted_rewards_per_episode)
        print(f'''reward baseline = {baseline}''')
        return baseline
    
    @cache()
    def gather_optimal_parameters(self, baseline):
        # 
        # hone in on acceptable epsilon
        # 
        print("----- finding optimal epsilon -------------------------------------------------------------------------------------------------------------------------------------")
        acceptable_performance_loss = baseline.stdev * self.settings.acceptable_performance_loss
        optimal_epsilon = self.settings.inital_epsilon
        optimal_horizon = self.settings.intial_horizon
        threshold_guesses = []
        failure_point_summaries = []
        for episode_index in range(self.settings.number_of_episodes_for_optimal_parameters):
            
            # per-timestep data
            forecast, rewards, discounted_rewards, failure_points, stopped_earlies, real_q_values, q_value_gaps = self.experience_episode(scaled_epsilon=optimal_epsilon, horizon=optimal_horizon, episode_index=episode_index)
            failure_point_summaries.append(simple_stats(failure_points))
            optimal_horizon = max((each.max for each in failure_point_summaries)) * 2 # e.g. make sure the horizon is plenty big, as it shouldn't be restricting the capabilites of prediciton
            # TODO: probably should increase the number of loops whenever horizon seems to have not converged (e.g. prediction length is growing)
            
            # 
            # adjust the optimal_epsilon up or down
            # 
            # technique
            #      if 1 standard devation below baseline, then keep as-is
            #      if equal to baseline, increase by ((this+stdev)/baseline)
            #      if 2 standard devations below baseline, decrease by ((this+stdev)/baseline)
            #     (^ smooth those changes with averaging instead of directly using them)
            episode_raw_score = sum(discounted_rewards)
            effective_score = episode_raw_score + acceptable_performance_loss
            adjustment_scale = effective_score / baseline.average
            threshold_guesses.append(adjustment_scale * optimal_epsilon)
            optimal_epsilon = average([optimal_epsilon, adjustment_scale * optimal_epsilon]) # smooth via simple averaging (may need to increase this smoother to be average(threshold_guesses))
            
            # 
            # logging
            # 
            print(f'''    episode={episode_index}, horizon={optimal_horizon}, effective_score={effective_score:.2f}, baseline={baseline.average:.2f}+{baseline.stdev:.2f}={baseline.stdev+baseline.average:.2f}, new_epsilon={optimal_epsilon:.4f}, adjustment%={(adjustment_scale-1)*100:.2f},''')
        
        print(f'''optimal_epsilon = {optimal_epsilon}''')
        print(f'''optimal_horizon = {optimal_horizon}''')
        return optimal_epsilon, optimal_horizon

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
        predictor.agent.gamma = config.train_agent.env_overrides[config.env_name].reward_discount
        
        # define return value
        self.csv_data = LazyDict(
            epsilon=[],
            rewards=[],
            discounted_rewards=[],
            forecast=[],
        )
        
        # get a baseline for the reward value
        baseline = self.gather_baseline()
        optimal_epsilon, optimal_horizon = self.gather_optimal_parameters(baseline)
        
        # save to a place they'll be easily visible
        horizon        = self.simple_data.optimal_epsilon = optimal_epsilon
        scaled_epsilon = self.simple_data.optimal_horizon = optimal_horizon
            
        # 
        # perform experiments with optimal
        # 
        epoch_recorder = Recorder(
            horizon=horizon,
            scaled_epsilon=scaled_epsilon,
        ).set_parent(self.recorder)
        
        index = -1
        forecast_slices = []
        alt_forecast_slices = []
        for episode_index in range(settings.number_of_episodes_for_testing):
            index += 1
            forecast, rewards, discounted_rewards, failure_points, stopped_earlies, real_q_values, q_value_gaps = self.experience_episode(scaled_epsilon=scaled_epsilon, horizon=horizon, episode_index=episode_index)
            
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
                **key_prepend("q_gaps", simple_stats(flatten(q_value_gaps) or [0])),
                **key_prepend("q_final_gaps", simple_stats([each[-1] for each in q_value_gaps if len(each) > 0] or [0])),
            )
            self.increment_live_graphs() # uses latest record
            
            # record for CSV backwards compatibility
            self.csv_data.epsilon.append(scaled_epsilon)
            self.csv_data.rewards.append(reward_stats.sum)
            self.csv_data.discounted_rewards.append(discounted_reward_stats.sum)
            self.csv_data.forecast.append(forecast[horizon:]) # BOOKMARK: len(forecast) == number_of_timesteps, so I have no idea why horizon is being used to slice it
            
            print(f"    scaled_epsilon: {scaled_epsilon:.4f}, forecast_average: {grand_forecast_average:.4f}, episode_reward:{reward_stats.sum:.2f}, max_timestep_reward: {reward_stats.max:.2f}, min_timestep_reward: {reward_stats.min:.2f}")
        
        self.save()
        return self

    # 
    # misc helpers
    # 
    def generate_graphs(self):
        smoothing = self.settings.graph_smoothing
        
        discounted_rewards       = []
        rewards                  = []
        q_values                 = []
        q_final_gaps_average     = []
        q_gaps_average           = []
        q_gaps_min               = []
        q_gaps_max               = []
        scaled_epsilons          = []
        forecasts_average        = []
        failure_points_average   = []
        horizons                 = []
        
        # for each epsilon-horizon pair
        for each_recorder in self.recorder.sub_recorders:
            discounted_rewards       += rolling_average(each_recorder.frame["discounted_reward_sum"]    , smoothing)
            rewards                  += rolling_average(each_recorder.frame["reward_sum"]               , smoothing)
            q_values                 += rolling_average(each_recorder.frame["q_average"]                , smoothing)
            q_final_gaps_average     += rolling_average(each_recorder.frame["q_final_gaps_average"]     , smoothing)
            q_gaps_average           += rolling_average(each_recorder.frame["q_gaps_average"]           , smoothing)
            q_gaps_min               += rolling_average(each_recorder.frame["q_gaps_min"]               , smoothing)
            q_gaps_max               += rolling_average(each_recorder.frame["q_gaps_max"]               , smoothing)
            forecasts_average        += rolling_average(each_recorder.frame["forecast_average"]         , smoothing)
            failure_points_average   += rolling_average(each_recorder.frame["failure_point_average"]    , smoothing)
            
            count = len(each_recorder.frame["reward_average"])
            scaled_epsilons += [ each_recorder["scaled_epsilon"] ]*count
            horizons        += [ each_recorder["horizon"]        ]*count
        
        # add indicies to all of them
        discounted_rewards       = tuple(enumerate(discounted_rewards       ))
        rewards                  = tuple(enumerate(rewards                  ))
        q_values                 = tuple(enumerate(q_values                 ))
        q_final_gaps_average     = tuple(enumerate(q_final_gaps_average     ))
        q_gaps_average           = tuple(enumerate(q_gaps_average           ))
        q_gaps_min               = tuple(enumerate(q_gaps_min               ))
        q_gaps_max               = tuple(enumerate(q_gaps_max               ))
        scaled_epsilons          = tuple(enumerate(scaled_epsilons          ))
        forecasts_average        = tuple(enumerate(forecasts_average        ))
        failure_points_average   = tuple(enumerate(failure_points_average   ))
        horizons                 = tuple(enumerate(horizons                 ))
        
        # 
        # display the actual cards
        # 
        reward_card = ss.DisplayCard("multiLine", dict(
            discounted_reward=discounted_rewards,
            reward=rewards,
        ))
        
        threshold_card = ss.DisplayCard("multiLine", dict(
            scaled_epsilon=scaled_epsilons,
            q_final_gaps_average=q_final_gaps_average,
            q_gaps_average=q_gaps_average,
            # q_gaps_min=q_gaps_min,
            # q_gaps_max=q_gaps_max,
            # timestep_q_average=q_values,
        ))
        prediction_card = ss.DisplayCard("multiLine", dict(
            forecast_average=forecasts_average,
            failure_point_average=failure_points_average,
            horizon=horizons,
        ))
        text_card = ss.DisplayCard("quickMarkdown", f"""## Experiment: {config.experiment_name}""")
        
        # 
        # save plots
        # 
        plot_kwargs = dict(
            csv_path=self.csv_path,
            output_folder=f"{self.path}/visuals",
            reward_discount=self.agent_reward_discount,
            min_reward_single_timestep=self.settings.min_reward_single_timestep,
            max_reward_single_timestep=self.settings.max_reward_single_timestep,
        )
        plot_epsilon_1(**plot_kwargs)
        plot_epsilon_2(**plot_kwargs)
        
        return self
    
    def init_live_graphs(self):
        self.reward_card = ss.DisplayCard("multiLine", dict(
            discounted_reward=[],
            reward=[],
        ))
        
        self.threshold_card = ss.DisplayCard("multiLine", dict(
            scaled_epsilon=[],
            q_final_gaps_average=[],
            q_gaps_average=[],
            # q_gaps_log_min=[],
            # q_gaps_log_max=[],
            # timestep_q_average=[],
        ))
        self.prediction_card = ss.DisplayCard("multiLine", dict(
            forecast_average=[],
            failure_point_average=[],
            horizon=[],
        ))
        self.text_card = ss.DisplayCard("quickMarkdown", f"""## Experiment: {config.experiment_name}""")
    
    def increment_live_graphs(self):
        if not self.threshold_card: self.init_live_graphs()
        
        index = sum(len(each) for each in self.recorder.sub_recorders)
        latest_record = self.recorder.sub_recorders[-1][-1]
        
        self.reward_card.send(dict(
            discounted_reward=[ index, latest_record["discounted_reward_sum"]],
            reward=           [ index, latest_record["reward_sum"]],
        ))
        self.threshold_card.send(dict(
            scaled_epsilon=          [ index,           latest_record["scaled_epsilon"]       ],
            q_final_gaps_average=    [ index,           latest_record["q_final_gaps_average"] ],
            q_gaps_average=          [ index,           latest_record["q_gaps_average"]       ],
            # q_gaps_log_min=          [ index, log_scale(latest_record["q_gaps_min"])          ],
            # q_gaps_log_max=          [ index, log_scale(latest_record["q_gaps_max"])          ],
            # timestep_q_average=      [index, latest_record["q_average"] ],
        ))
        self.prediction_card.send(dict(
            forecast_average=      [index, latest_record["forecast_average"] ],
            failure_point_average= [index, latest_record["failure_point_average"] ],
            horizon=               [index, latest_record["horizon"] ],
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
        "simple_data",
        "recorder",
        "agent_reward_discount",
        "rewards_per_episode_per_timestep",
        "discounted_rewards_per_episode_per_timestep",
        "failure_points_per_episode_per_timestep",
        "stopped_early_per_episode_per_timestep",
        "real_q_values_per_episode_per_timestep",
        "q_value_gaps_per_episode_per_timestep",
    ]
    
    @classmethod
    def smart_load(cls, path, settings, predictor, force_recompute=False):
        if not force_recompute and all(
            FS.is_file(f"{path}/serial_data/{each_attribute_name}.pickle")
                for each_attribute_name in cls.attributes_to_save
        ):
            return cls.load(
                path=path,
                settings=settings,
                predictor=predictor,
            ).generate_graphs()
        return Tester(
            path=path,
            settings=settings,
            predictor=predictor,
        ).run_all_episodes().generate_graphs()
        
    @classmethod
    def load(cls, path, settings={}, predictor=None):
        attributes = {}
        for each_attribute_name in cls.attributes_to_save:
            attributes[each_attribute_name] = large_pickle_load(f"{path}/serial_data/{each_attribute_name}.pickle")
        attributes["settings"].update(settings)
        # create a tester with the loaded data
        return Tester(
            settings=attributes["settings"],
            predictor=predictor,
            attribute_overrides=attributes,
            path=path,
        )
    
    def save(self, path=None):
        path = path or self.path
        # save normal things
        for each_attribute_name in self.attributes_to_save:
            each_path = f"{path}/serial_data/{each_attribute_name}.pickle"
            FS.clear_a_path_for(each_path, overwrite=True)
            large_pickle_save(getattr(self, each_attribute_name, None), each_path)
        
        # save basic data
        simple_data_path = f"{path}/simple_data.json"
        FS.clear_a_path_for(simple_data_path, overwrite=True)
        ez_yaml.to_file(obj=self.simple_data, file_path=simple_data_path)
        
        # save csv
        FS.clear_a_path_for(self.csv_path, overwrite=True)
        pd.DataFrame(self.csv_data).explode("forecast").to_csv(self.csv_path)
        return self



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

def plot_epsilon_1(csv_path, output_folder, max_reward_single_timestep, min_reward_single_timestep, reward_discount):
    if not FS.exists(csv_path):
        print(f"no data found for: {csv_path}")
        return
    
    data_frame = pd.read_csv(csv_path)
    
    max_reward_single_timestep = data_frame.groupby('epsilon').discounted_rewards.mean().values[0]
    score_range = max_reward_single_timestep - min_reward_single_timestep
    data_frame['normalized_rewards'] = (data_frame.discounted_rewards - min_reward_single_timestep) / score_range
    data_frame['epsilon_adjusted'] = data_frame.epsilon / score_range
    epsilon_means       = data_frame.groupby('epsilon_adjusted').normalized_rewards.mean()
    standard_deviations = data_frame.groupby('epsilon_adjusted').normalized_rewards.std().values
    epsilon_low         = data_frame.epsilon_adjusted.min()
    epsilon_high        = data_frame.epsilon_adjusted.max()
    epsilon             = np.linspace(0, epsilon_high, 2)
    subopt              = epsilon / (1 - reward_discount)
    base_mean           = 1
    
    plt.figure(figsize=(5.5, 5.5))
    plt.plot(epsilon, base_mean-subopt, color='orange', label='Perform. Bounds')
    epsilon_means.plot(label='V^{PAC}(s_0)', marker='o', ax=plt.subplot(1,1,1), color='steelblue')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Epsilon Coefficient')
    plt.fill_between(epsilon_means.index, epsilon_means-standard_deviations, epsilon_means+standard_deviations, alpha=0.2, color='steelblue')
    plt.hlines(base_mean, epsilon_low, epsilon_high, color='green', label='V^pi(s_0)')
    plt.plot([epsilon_low, epsilon_high], [0,0], color='red', label='V^{rand}(s_0)')
    plt.legend(loc='lower left', fontsize="x-small")
    plt.tight_layout()
    
    plt.savefig(
        FS.clear_a_path_for(
            f"{output_folder}/epsilon_1",
            overwrite=True
        )
    )

def plot_epsilon_2(csv_path, output_folder, max_reward_single_timestep, min_reward_single_timestep, reward_discount):
    if not FS.exists(csv_path):
        print(f"no data found for: {csv_path}")
        return
    
    data_frame = pd.read_csv(csv_path).rename(columns={'Unnamed: 0': 'episode'})
    score_range = max_reward_single_timestep - min_reward_single_timestep
    data_frame['normalized_rewards'] = (data_frame.discounted_rewards - min_reward_single_timestep) / score_range
    data_frame['epsilon_adjusted']   = data_frame.epsilon / score_range
    forecast_means      = data_frame.groupby('epsilon_adjusted').forecast.mean() + 1
    standard_deviations = data_frame.groupby(['epsilon_adjusted', 'episode']).forecast.mean().unstack().std(1)
    forecast_low  = forecast_means + standard_deviations
    forecast_high = forecast_means - standard_deviations
    
    plt.figure(figsize=(5.5, 5.5))
    forecast_means.plot(marker='o', label='V^F(s_0)', color='steelblue')
    plt.fill_between(forecast_means.index, np.where(forecast_low < 0, 0, forecast_low), forecast_high, alpha=0.2, color='steelblue')
    plt.xlabel('Epsilon Coefficient')
    plt.ylabel('Forecast')
    plt.savefig(
        FS.clear_a_path_for(
            f"{output_folder}/epsilon_2",
            overwrite=True
        )
    )