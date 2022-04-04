import os
from posixpath import basename

import gym
import numpy as np
import stable_baselines3 as sb
import torch
from torch import nn
from torch.optim import Adam
from trivial_torch_tools import to_tensor, Sequential, init, convert_each_arg
from trivial_torch_tools.generics import to_pure, large_pickle_save, large_pickle_load
from super_map import LazyDict
import file_system_py as FS
from stable_baselines3 import SAC as BasicSAC

from info import path_to, config
from main.training.basic_agent import Agent
from main.training.dynamics import DynamicsModel
from main.tools import flatten, get_discounted_rewards, divide_chunks, minibatch, ft, Episode, train_test_split, TimestepSeries, Timestep, to_numpy, feed_forward
from main.agents.agent_skeleton import Skeleton

minibatch_size = config.train_dynamics.minibatch_size

class SAC(BasicSAC):
    @classmethod
    def load_for(cls, path, env_name):
        SAC.load(
            path,
            env=env_name,
            device=config.device,
        )
        
    @init.freeze_tools()
    def __init__(self, env, path=None, **kwargs):
        super().__init__(policy="MlpPolicy", env=env, device=config.device, verbose=2, **kwargs)
        self.path = path
        self.env = env
        # policy:          Union[str, Type[stable_baselines3.sac.policies.SACPolicy]],
        # env:             Union[gym.core.Env, stable_baselines3.common.vec_env.base_vec_env.VecEnv, str],
        # learning_rate:   Union[float, Callable[[float], float]] = 0.0003,
        # buffer_size:     int = 1000000,
        # learning_starts: int = 100,
        # batch_size:      int = 256,
        # tau:             float = 0.005,
        # gamma:           float = 0.99,
        # train_freq:      Union[int, Tuple[int, str]] = 1,
        # gradient_steps:  int = 1,
    
    # this is for freeze()
    def children(self):
        return [self.critic, self.actor]
    
    def load(self, path):
        path = path or self.path
        return SAC.load(
            path,
            self.env,
            device=config.device,
        )

# 
# helper model
# 
class CoachModel(nn.Module):
    @init.to_device()
    @init.freeze_tools()
    @init.save_and_load_methods(model_attributes=["model"], basic_attributes=[ "hidden_sizes", "learning_rate", "obs_dim", "act_dim"])
    def __init__(self, obs_dim, act_dim, hidden_sizes, learning_rate, agent, **config):
        super().__init__()
        self.act_dim       = act_dim
        self.obs_dim       = obs_dim
        self.learning_rate = learning_rate
        self.config        = config
        self.model         = feed_forward(
            layer_sizes=[obs_dim + act_dim, *hidden_sizes, obs_dim],
            activation=nn.ReLU
        )
    
    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="hardware")
    def forward(self, state_batch, action_batch):
        return self.model.forwards(torch.cat((state_batch, action_batch), -1))

    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="hardware")
    def predict(self, obs: np.ndarray, act: np.ndarray):
        with torch.no_grad():
            next_observation = self.model(torch.cat((obs, act), -1))
        return to_numpy(next_observation)


# 
# 
# Agent
# 
# 
class PredictiveAgent(Skeleton):
    basic_attributes=['env_name', 'coach_config', 'sac_config', 'config']
    
    @classmethod
    def sub_paths_for(cls, *, path):
        normal_data_path = path+"/normal_data.pickle"
        coach_path = path+"/coach.model"
        sac_path = path+"/sac.model"
        return normal_data_path, coach_path, sac_path
    
    @classmethod
    def load(cls, *, path):
        self = LazyDict()
        self.path = path
        normal_data_path, coach_path, sac_path = PredictiveAgent.sub_paths_for(path)
        
        # 
        # basics
        # 
        normal_data = large_pickle_load(normal_data_path)
        for each_attribute, each_value in zip(basic_attributes, normal_data):
            setattr(self, each_attribute, each_value)
        # 
        # env
        # 
        env = config.get_env(self.env_name)
        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space
        
        # 
        # coach
        # 
        self.coach = CoachModel(
            obs_dim=observation_space.shape[0],
            act_dim=action_space.shape[0],
            hidden_sizes=self.coach_config.hidden_sizes,
            learning_rate=self.coach_config.learning_rate,
            **coach_config,
        )
        self.coach.load_state_dict(torch.load(coach_path))
        
        # 
        # SAC
        # 
        self.sac = SAC.load_for(sac_path, self.env_name)
        
        PredictiveAgent(**self)
    
    def save(self, path):
        normal_data_path, coach_path, sac_path = PredictiveAgent.sub_paths_for(path)
        # 
        # save normal data
        # 
        normal_data = tuple(getattr(self, each_attribute) for each_attribute in PredictiveAgent.basic_attributes)
        large_pickle_save(normal_data, normal_data)
        # 
        # save coach
        # 
        torch.save(self.coach.state_dict(), coach_path)
        large_pickle_save(normal_data, normal_data)
        # 
        # save sac
        # 
        self.sac.save(sac_path)    
        
    # 
    # init
    # 
    @init.to_device()
    def __init__(self, *, env_name, observation_space, action_space, coach, sac, coach_config, sac_config, path=None, **kwargs):
        super().__init__()
        self.env_name     = env_name
        self.coach_config = coach_config
        self.sac_config   = sac_config
        self.config       = kwargs
        
        # other setup
        self.hardware = config.device
        
        # from Skeleton
        self.observation = None
        self.reward = None
        self.action = None
        self.episode_is_over = None
        # for training
        self.time_series = None
        self.current_timestep = Timestep()
    
    def when_mission_starts(self, mission_index=0):
        self.horizon = config.train_predictive.initial_horizon_size
        self.loss_threshold = config.train_predictive.loss_threshold
        self.record = LazyDict(
            actor_losses=[],
            critic_losses=[],
            coach_losses=[],
            horizons=[],
        )
    
    def when_episode_starts(self, episode_index):
        self.time_series = TimestepSeries()
    
    def when_timestep_starts(self, timestep_index):
        self.current_timestep = Timestep()
        self.current_timestep.index = timestep_index
        self.current_timestep.prev_state = self.observation
        self.current_timestep.action = self.action = self.make_decision(self.observation)
    
    def when_timestep_ends(self, timestep_index):
        self.current_timestep.reward = self.reward
        self.current_timestep.state = self.observation
        self.time_series.add(self.current_timestep)
    
    def when_episode_ends(self, episode_index):
        pass
    
    def when_mission_ends(self, mission_index=0):
        pass
    
    # 
    # 
    # capabilities
    # 
    # 
    
    # Actor
    def make_decision(self, observation_batch, deterministic=True):
        observation_batch = to_tensor(observation_batch).to(self.device)
        action_batch = self.sac.actor.forward(observation_batch, deterministic=deterministic)
        return action_batch
    
    # Critic
    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device()
    @convert_each_arg.to_batched_tensor(number_of_dimensions=2)
    def value_of(self, state, action):
        """
            batched:
                state.shape = torch.Size([32, 2])
                action.shape = torch.Size([32, 1])
        """
        result = self.sac.critic_target(state, action)
        q = torch.cat(result, dim=1)
        q, _ = torch.min(q, dim=1, keepdim=True)
        return q
    
    # Coach
    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device()
    def predict_state(self, state_batch, action_batch):
        return self.coach.model.forwards(torch.cat((state_batch, action_batch), -1))
    
    
    # 
    # losses
    # 