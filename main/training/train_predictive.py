import os
from types import FunctionType
from typing import List
from copy import deepcopy

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
from simple_namespace import namespace

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

loss_batch = []
def replan(
    intial_state: NDArray,
    old_plan: NDArray,
    forecast: List[int],
    epsilon: float,
    forecast_horizon: int,
    action_size: int,
    agent: OffPolicyAlgorithm,
    dynamics: DynamicsModel,
    actor_copy,
    actor_copy_optimizer,
):
    global loss_batch
    
    # 
    # helpers
    # 
    def get_actor_action_for(state):
        action, _ = agent.predict(state, deterministic=True)
        return action 
    
    def get_actor_copy_action_for(state):
        action = actor_copy(to_tensor([state]))[0]
        return action 
    
    def without_grad_get_value(state, action):
        with torch.no_grad():
            return agent.value_of(state, action)
    
    def with_grad_get_value(state, action):
        return agent.value_of(state, action)
    
    def run_backprop_if_ready():
        if len(loss_batch) >= config.train_predictive.weight_update_frequency:
            actor_copy_optimizer.zero_grad()
            loss = torch.stack(loss_batch).mean()
            loss.backward()
            actor_copy_optimizer.step()
            loss_batch = []
    
    # 
    # runtime
    # 
    new_plan = []

    state = intial_state
    future_plan = old_plan[1:] # reuse old plan (recycle)
    for forcast_index, (predicted_state, planned_action) in enumerate(future_plan):
        forecast[forcast_index] = forecast[forcast_index+1] + 1 # for the stats... keep track of the forecast of this action
        
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
        # BOOKMARK: check this, seems like it could be problematic (expanded from DynamicsModel.forward)
        with torch.no_grad():
            # the likely reason this no_grad is here is 
            # so that future states can be predicted 
            # without penalizing the first one for +2 timestep-loss, +3 timestep loss, +4 timestep loss, etc
            # just generate the state, then redo +1 timestep-loss only
            state = dynamics.model(torch.cat((state, plan_action), -1)).cpu().numpy()
        
    # produce new plan (replan)
    for index in range(forcast_index, forecast_horizon):
        action = get_actor_copy_action_for(state)
        new_plan.append((state, action))
        
        with torch.no_grad():
            state = dynamics.model(torch.cat((state, action), -1)).cpu().numpy()
        # Below is what used to be above
        # with torch.no_grad(): # this is a double-no-grad cause there's already a no-grad inside dynamics.forward()
        #     state = dynamics(state, action)
        
        forecast[index] = 0

    return new_plan, forecast


def experience(
    epsilon: float,
    forecast_horizon: int,
    action_size: int,
    agent: OffPolicyAlgorithm,
    actor_copy,
    dynamics: DynamicsModel,
    env,
    actor_copy_optimizer,
):
    episode_forecast = []
    rewards          = []
    empty_plan       = []
    zero_forecasts   = np.zeros(forecast_horizon, np.int8)
    
    state = env.reset()
    plan, forecasts = replan(
        state,
        empty_plan,
        zero_forecasts,
        epsilon,
        forecast_horizon,
        action_size,
        agent,
        actor_copy,
        dynamics,
        actor_copy_optimizer,
    )
    done = False
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
            actor_copy,
            dynamics,
            actor_copy_optimizer,
        )
    return rewards, episode_forecast

def test_afrl(
    epsilons: List[float],
    forecast_horizon: int,
    action_size: int,
    agent: OffPolicyAlgorithm,
    actor_copy,
    dynamics: DynamicsModel,
    n_experiments: int,
    env,
    actor_copy_optimizer,
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
                actor_copy,
                dynamics,
                env,
                actor_copy_optimizer,
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
    actor_copy           = deepcopy(agent.policy)
    actor_copy_optimizer = Adam(actor_copy.parameters(), lr=0.0001) # QUESTION: why does the actor need an actor_copy_optimizer?

    # actor_copy.load_state_dict(torch.load(f'data/models/agents/actor_copy/{env_name}.pt'))

    print("Gamma:", agent.gamma)

    return test_afrl(
        epsilons,
        forecast_horizon,
        action_size,
        agent,
        actor_copy,
        dynamics,
        n_experiments,
        env,
        actor_copy_optimizer,
    )

if __name__ == "__main__":
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
