import functools
from dataclasses import dataclass
from types import *
from random import random, sample, choices, shuffle
from collections import defaultdict

import gym
import numpy as np
import stable_baselines3 as sb
import torch
import file_system_py as FS
import silver_spectacle as ss
from torch import nn, Tensor
from torch.optim import Adam
from rigorous_recorder import Recorder
from trivial_torch_tools import to_tensor, Sequential, init, convert_each_arg
from trivial_torch_tools.generics import large_pickle_load, large_pickle_save, to_pure
from super_map import LazyDict

from debug import debug
from info import path_to, config
from tools import flatten, get_discounted_rewards, divide_chunks, minibatch, ft, Episode, train_test_split, TimestepSeries, to_numpy, feed_forward, bundle, average, log_graph, WeightUpdate
from main.agent import Agent

settings = config.train_coach
# contains:
    # learning_rate
    # hidden_sizes
    # number_of_episodes
    # number_of_epochs
    # minibatch_size
    # etc

# State transition coach model
class Coach(nn.Module):
    """
    The model of how the world works
        (state) => (next_state)
    """
    
    basic_attributes = [ "settings", "learning_rate", "which_loss", "hidden_sizes", "state_size", "action_size" ]
    
    # 
    # load
    # 
    @classmethod
    def smart_load(cls,
        env_name,
        *,
        path,
        agent,
        force_retrain=settings.force_retrain,
    ):
        env = config.get_env(env_name)
        coach = Coach(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.shape[0],
            settings=settings.merge(settings.env_overrides.get(env_name, {})),
            agent=agent,
            path=path,
            device=config.device,
        )
        # load if exists
        if not force_retrain:
            if FS.exists(path):
                print(f'''\n\n-----------------------------------------------------------------------------------------------------''')
                print(f''' Coach Model Exists, loading: {path}''')
                print(f'''-----------------------------------------------------------------------------------------------------\n\n''')
                path_to = Coach.internal_paths(path)
                coach.load_state_dict(torch.load(path_to.model))
                coach.recorder = Recorder.load_from(path_to.recorder)
                return coach
        
        print(f'''\n\n-----------------------------------------------------------------------------------------------------''')
        print(f''' Training Coach Model from scrach''')
        print(f'''-----------------------------------------------------------------------------------------------------\n\n''')
        # 
        # train
        # 
        coach.train_with(
            env_name,
            agent=agent,
            number_of_episodes=settings.number_of_episodes,
            number_of_epochs=settings.number_of_epochs,
            with_card=settings.with_card,
            minibatch_size=settings.minibatch_size,
        )
        
        # 
        # save
        # 
        coach.save(path)
        if settings.with_card: coach.generate_graphs()
        return coach
    
    # init
    def __init__(self, state_size, action_size, settings, device, agent, path, **kwargs):
        super().__init__()
        self.settings      = settings
        self.learning_rate = self.settings.learning_rate
        self.which_loss    = self.settings.loss_function
        self.hidden_sizes  = self.settings.hidden_sizes
        self.device        = device
        self.state_size       = state_size
        self.action_size       = action_size
        self.agent         = agent
        self.path          = path
        self.recorder      = Recorder(
            experiment_name=config.experiment_name,
            model="coach",
        )
        self.episode_recorder = None
        self.model = feed_forward(layer_sizes=[state_size + action_size, *self.hidden_sizes, state_size], activation=nn.ReLU).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_objects = LazyDict({
            each.__name__ : each()
                for each in [
                    self.consistent_value_loss,
                    self.consistent_coach_loss,
                    self.value_prediction_loss,
                    self.action_prediction_loss,
                    self.state_prediction_loss,
                ]
        })
        
    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="device")
    def forward(self, state_batch, action_batch):
        return self.model.forward(torch.cat((state_batch, action_batch), -1))

    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="device")
    def predict(self, obs: np.ndarray, act: np.ndarray):
        with torch.no_grad():
            next_observation = self.model(torch.cat((obs, act), -1))
        return to_numpy(next_observation)
    
    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="device")
    def create_forcast(self, observation: torch.Tensor, inital_action: torch.Tensor, length: int):
        """
        Return:
            list of state-action pairs
            all pairs are predictive
        """
        predictions = []
        state = observation
        action = inital_action
        with self.agent.frozen() as agent:
            for each in range(length):
                predicted_state = self.forward(state, action)
                predicted_action = agent.make_decision(state)
                predictions.append(tuple(predicted_state, predicted_action))
                state = predicted_state
                action = predicted_action
        return predictions
    
    
    # 
    # Loss Helpers
    # 
    def create_next_batch(self, indices, minibatch_size, lookahead, data):
        sideways_batch = []
        for count in range(minibatch_size):
            index = indices.pop()
            try:
                items = []
                for each_offset in range(lookahead+1):
                    for each_data_source in data:
                        items.append(each_data_source[index+each_offset])
                sideways_batch.append(tuple(items))
            # skip any index where lookahead isnt possible
            except IndexError as error:
                pass
        
        # for example return: "state_1s, action_1s, state_2s, action_2s" each as a tensor
        return tuple(to_tensor(each) for each in zip(*sideways_batch))
                    
    
    # 
    # Loss functions
    # 
    def consistent_value_loss(self):
        output = LossObject()
        
        def actual_loss_function(state_1s, action_1s, state_2s, action_2s, state_3s, action_3s, *_):
            once_predicted_state_2s  = self.forward(state_1s, action_1s)
            once_predicted_state_3s  = self.forward(state_2s, action_2s)
            twice_predicted_state_3s = self.forward(once_predicted_state_2s, action_2s)
            
            actual_value    = self.agent.value_of(state_3s, action_3s)
            predicted_once  = self.agent.value_of(once_predicted_state_3s, action_3s)
            predicted_twice = self.agent.value_of(twice_predicted_state_3s, action_3s)
            
            output.loss_value = torch.stack([
                ((actual_value   - predicted_once )**2).mean(dim=-1),
                ((predicted_once - predicted_twice)**2).mean(dim=-1),
            ]).mean()
            
            return output.loss_value
        
        output.lookahead = 2 # "state_2s, action_2" is 1-ahead,  "state_3s, action_3s" is 2-ahead
        output.function = actual_loss_function
        return output
    
    def consistent_coach_loss(self):
        output = LossObject()
        
        state_prediction_loss = self.state_prediction_loss().function
        value_prediction_loss = self.value_prediction_loss().function
        scale_future_state_loss = self.settings.consistent_coach_loss.scale_future_state_loss
        def actual_loss_function(state_1s, action_1s, state_2s, action_2s, state_3s, action_3s, *_):
            once_predicted_state_2s  = self.forward(state_1s, action_1s)
            once_predicted_state_3s  = self.forward(state_2s, action_2s)
            twice_predicted_state_3s = self.forward(once_predicted_state_2s, action_2s)
            
            future_loss = ((once_predicted_state_3s - twice_predicted_state_3s)**2).mean() * scale_future_state_loss
            q_loss     = value_prediction_loss(state_1s, action_1s, state_2s, action_2s, state_3s, action_3s, *_)
            state_loss = state_prediction_loss(state_1s, action_1s, state_2s, action_2s, state_3s, action_3s, *_)
            # BOOKMARK
            output.loss_value = future_loss + q_loss
            return output.loss_value
        
        output.lookahead = 2 # "state_2s, action_2" is 1-ahead,  "state_3s, action_3s" is 2-ahead
        output.function = actual_loss_function
        return output
    
    def value_prediction_loss(self):
        output = LossObject()
            
        def actual_loss_function(state_1s, action_1s, state_2s, action_2s, *_):
            predicted_state_2s   = self.forward(state_1s, action_1s)
            
            predicted_action_2s = self.agent.make_decision(predicted_state_2s, deterministic=True)
            predicted_value_2s  = self.agent.value_of(state_2s, predicted_action_2s)
            best_value_2s       = self.agent.value_of(state_2s, action_2s)
            
            return ((best_value_2s - predicted_value_2s)**2).mean()
            
        output.lookahead = 1 # "state_2s, action_2" is 1-ahead,  "state_3s, action_3s" is 2-ahead
        output.function = actual_loss_function
        return output
    
    def action_prediction_loss(self):
        output = LossObject()
            
        def actual_loss_function(state_1s, action_1s, state_2s, action_2s, *_):
            predicted_state_2s  = self.forward(state_1s, action_1s)
            predicted_action_2s = self.agent.make_decision(predicted_state_2s, deterministic=True)
            
            output.loss_value = ((action_2s - predicted_action_2s) ** 2).mean() # when action is very different, loss is high
            
            return output.loss_value
            
        output.lookahead = 1 # "state_2s, action_2" is 1-ahead,  "state_3s, action_3s" is 2-ahead
        output.function = actual_loss_function
        return output
        
        
    def state_prediction_loss(self):
        output = LossObject()
            
        value_prediction_loss = self.value_prediction_loss().function
        def actual_loss_function(state_1s, action_1s, state_2s, action_2s, *_):
            predicted_state_2s = self.forward(state_1s, action_1s)
            
            q_error = value_prediction_loss(state_1s, action_1s, state_2s, action_2s, *_)
            output.loss_value = ((predicted_state_2s - state_2s) ** 2).mean()
            return output.loss_value
            
        output.lookahead = 1 # "state_2s, action_2" is 1-ahead,  "state_3s, action_3s" is 2-ahead
        output.function = actual_loss_function
        return output
    
    # 
    # 
    # main training loop
    # 
    # 
    def train_with(self, env_name, agent, number_of_episodes=100, number_of_epochs=100, with_card=True, minibatch_size=settings.minibatch_size):
        env      = config.get_env(env_name)
        card     = None if not with_card else ss.DisplayCard("multiLine", {
            **{ f"train_{name}": [] for name in self.loss_objects },
            **{ f"test_{name}" : [] for name in self.loss_objects },
        })
        self.episode_recorder = Recorder(
            training_record=True,
            env_name=env_name,
            batch_size=minibatch_size,
            number_of_episodes=number_of_episodes,
            number_of_epochs=number_of_epochs,
        ).set_parent(self.recorder)
    
        # Get experience from trained agent
        episodes, all_actions, all_curr_states = agent.gather_experience(env, number_of_episodes)
        print(f'''Starting Train/Test''')
        states      = to_tensor(all_curr_states)
        actions     = to_tensor(all_actions)

        (
            (train_states , test_states),
            (train_actions, test_actions),
        ) = train_test_split(
            states,
            actions,
            split_proportion=settings.train_test_split,
        )
        # send to cuda if needed
        train_data = train_states.to(self.device), train_actions.to(self.device)
        test_data  = test_states.to(self.device) , test_actions.to(self.device)
        
        with self.agent.frozen:
            # 
            # epochs
            # 
            for epochs_index in range(number_of_epochs):
                self.episode_recorder.add(
                    epochs_index=epochs_index,
                )
                # 
                # 
                # training
                # 
                # 
                self.train(True)
                indices = list(range(len(train_data[0])))
                per_batch_data = defaultdict(lambda *_: [])
                shuffle(indices)
                while len(indices) > minibatch_size:
                    # dynamicly shape batch dependong on lookahead
                    batch = self.create_next_batch(
                        indices=indices,
                        minibatch_size=minibatch_size,
                        lookahead=max(each_loss_object.lookahead for each_loss_object in self.loss_objects.values()),
                        data=train_data,
                    )
                    
                    #
                    # run minibatch
                    #
                    with WeightUpdate(optimizer=self.optimizer) as backprop:
                        # run all of them for logging purposes
                        for name, loss_object in self.loss_objects.items():
                            loss = loss_object.function(*batch)
                            
                            # if its the main/selected loss, backpropogate the output
                            if name == self.which_loss:
                                backprop.loss = loss
                            
                            # log data
                            per_batch_data["train_"+name].append(to_pure(loss))
                
                # record the averge for each loss function
                self.episode_recorder.add({
                    each_key: average(batch_values)
                        for each_key, batch_values in per_batch_data.items()
                            if len(batch_values) > 0
                })
              
                # 
                # testing
                # 
                self.train(False)
                # one giant batch
                batch = self.create_next_batch(
                    indices=list(range(len(test_data[0]))),
                    minibatch_size=len(test_data[0]),
                    lookahead=max(each_loss_object.lookahead for each_loss_object in self.loss_objects.values()),
                    data=test_data,
                )
                for name, loss_object in self.loss_objects.items():
                    self.episode_recorder.add({
                        "test_"+name : to_pure( loss_object.function(*batch) ) 
                    })
                
                
                # 
                # wrap up
                # 
                self.episode_recorder.push() # commits data record
                # graph all the losses
                if card: card.send({
                    each_key: [ epochs_index, each_value ]
                        for each_key, each_value in self.episode_recorder[-1].items()
                            if "loss" in each_key
                })
                
                print(f'''    Epoch: {epochs_index}''')
                
        return self.episode_recorder
    
    def generate_graphs(self):
        
        # y axis of loss
        # add the state_loss 
        # add the future_loss
        # add the q_value_loss
        records = tuple(self.recorder.all_records)
        training_records = tuple(each for each in records if each.get("training_record", False))
        special_records = tuple(each for each in records if each.get("future_loss", False) )
        ss.DisplayCard("multiLine", dict(
            train=[ (each["epochs_index"], each["train_loss"]) for each in training_records ],
            test= [ (each["epochs_index"], each["test_loss"] ) for each in training_records ],
            
            future_loss= [ (each["epochs_index"], each["future_loss"] ) for each in special_records ],
            q_loss= [ (each["epochs_index"], each["q_loss"] ) for each in special_records ],
            state_loss= [ (each["epochs_index"], each["state_loss"] ) for each in special_records ],
        ))
        return self
    
    @classmethod
    def internal_paths(cls, path):
        return LazyDict(
            model      = f"{path}/model.pt",
            attributes = f"{path}/attributes.pickle",
            recorder   = f"{path}/recorder.pickle",
        )
    
    def save(self, path=None):
        path = path or self.path
        print(f'''Saving Coach to: {path}''')
        path_to = Coach.internal_paths(path)
        
        basic_attribute_data = { each_attribute: getattr(self, each_attribute) for each_attribute in self.basic_attributes }
        
        large_pickle_save(basic_attribute_data, path_to.attributes)
        torch.save(self.state_dict(), FS.clear_a_path_for(path_to.model, overwrite=True))
        self.recorder.save_to(FS.clear_a_path_for(path_to.recorder, overwrite=True))
    
    @classmethod
    def load(cls, path, agent, device):
        path_to = Coach.internal_paths(path)
        
        basic_attribute_data = LazyDict(large_pickle_load(path_to.attributes))
        coach = Coach(
            state_size=basic_attribute_data.state_size,
            action_size=basic_attribute_data.action_size,
            settings=basic_attribute_data.settings,
            agent=agent,
            path=path,
            device=device,
        )
        coach.load_state_dict(torch.load(path_to.model))
        coach.recorder = Recorder.load_from(path_to.recorder)
        return coach

@dataclass
class LossObject:
    function: FunctionType = lambda : 0  # the actual loss function
    loss_value: Tensor = None      # optional, the most-recently calculated loss value
    lookahead: int = 0
    # "state, action" when lookahead = 0
    # "state, action, next_state, next_actoin" when lookahead = 1
    # "state_1s, action_1s, state_2s, action_2s, state_3s, action_3s" when lookahead = 2
    # etc