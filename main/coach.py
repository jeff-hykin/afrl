import functools

import gym
import numpy as np
import stable_baselines3 as sb
import torch
import file_system_py as FS
import silver_spectacle as ss
from torch import nn
from torch.optim import Adam
from trivial_torch_tools import to_tensor, Sequential, init, convert_each_arg
from trivial_torch_tools.generics import to_pure
from rigorous_recorder import RecordKeeper
from trivial_torch_tools.generics import large_pickle_load, large_pickle_save
from super_map import LazyDict

from debug import debug
from info import path_to, config
from tools import flatten, get_discounted_rewards, divide_chunks, minibatch, ft, Episode, train_test_split, TimestepSeries, to_numpy, feed_forward, bundle
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
        self.recorder      = RecordKeeper(
            experiment_name=config.experiment_name,
            model="coach",
        )
        self.model = feed_forward(layer_sizes=[state_size + action_size, *self.hidden_sizes, state_size], activation=nn.ReLU).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.losses = []
        
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
    # Loss Application (boilerplate code)
    # 
    def timestep_testing_loss(self, indices: list, step_data: tuple):
        loss_function = getattr(self, self.which_loss)
        loss = loss_function(indices, step_data)
        return to_pure(loss)
    
    def timestep_training_loss(self, indices: list, step_data: tuple):
        loss_function = getattr(self, self.which_loss)
        
        self.agent.freeze()
        loss = loss_function(indices, step_data)
        
        # Optimize the self model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.agent.unfreeze()
        return loss
    
    # 
    # Loss functions
    # 
    def consistent_value_loss(self, indices: list, step_data: tuple):
        states, actions = step_data
        start = min(indicies)
        end   = max(indices)
        
        # sliders
        #  [1,2,3,4,5]
        #  =>
        #  [1,2,3]
        #    [2,3,4]
        #      [3,4,5]
        state_1s, action_1s = states[start+0:end-2], actions[start+0:end-2]
        state_2s, action_2s = states[start+1:end-1], actions[start+1:end-1]
        state_3s, action_3s = states[start+2:end  ], actions[start+2:end  ]
        
        # action(#) is the action taken when state(#) is given to the actor
        
        once_predicted_state_2s  = self.forward(state_1s, action_1s)
        once_predicted_state_3s  = self.forward(state_2s, action_2s)
        twice_predicted_state_3s = self.forward(once_predicted_state_2s, action_2s)
        
        actual_value    = self.agent.value_of(state_3s, action_3s)
        predicted_once  = self.agent.value_of(once_predicted_state_3s, action_3s)
        predicted_twice = self.agent.value_of(twice_predicted_state_3s, action_3s)
        
        return torch.stack([
            ((actual_value   - predicted_once )**2).mean(dim=-1),
            ((predicted_once - predicted_twice)**2).mean(dim=-1),
        ]).mean()
    
    def consistent_coach_loss(self, indices: list, step_data: tuple):
        scale_value_prediction = self.settings.consistent_coach_loss.scale_value_prediction
        states, actions = step_data
        start = min(indicies)
        end   = max(indices)
        
        # sliders
        #  [1,2,3,4,5]
        #  =>
        #  [1,2,3]
        #    [2,3,4]
        #      [3,4,5]
        state_1s, action_1s = states[start+0:end-2], actions[start+0:end-2]
        state_2s, action_2s = states[start+1:end-1], actions[start+1:end-1]
        state_3s, action_3s = states[start+2:end  ], actions[start+2:end  ]
        
        once_predicted_state_2s = self.forward(state_1s, action_1s)
        once_predicted_state_3s = self.forward(state_2s, action_2s)
        twice_predicted_state_3s = self.forward(once_predicted_state_2s, action_2s)
        future_loss = ((once_predicted_state_3s - twice_predicted_state_3s)**2).mean(dim=-1)
        value_prediction_losses = self.value_prediction_loss(indices, step_data) * scale_value_prediction
        return (future_loss.sum() + value_prediction_losses.sum())/(future_loss.shape[0] + value_prediction_losses.shape[0])
    
    def value_prediction_loss(self, indices: list, step_data: tuple):
        states, actions = step_data
        start = min(indicies)
        end   = max(indices)
        states     , actions      = states[start+0:end-1], actions[start+0:end-1]
        next_states, next_actions = states[start+1:end  ], actions[start+1:end  ]
        
        predicted_next_states   = self.forward(states, actions)
        
        predicted_next_actions = self.agent.make_decision(predicted_next_states, deterministic=True)
        predicted_next_value   = self.agent.value_of(next_states, predicted_next_actions)
        best_next_value        = self.agent.value_of(next_states, next_actions)
        
        return (best_next_value - predicted_next_value).mean(dim=-1) # when predicted_next_value is high, loss is low (negative)
    
    def action_prediction_loss(self, indices: list, step_data: tuple):
        states, actions = step_data
        start = min(indicies)
        end   = max(indices)
        states     , actions      = states[start+0:end-1], actions[start+0:end-1]
        next_states, next_actions = states[start+1:end  ], actions[start+1:end  ]
        
        predicted_next_states  = self.forward(states, actions)
        predicted_next_actions = self.agent.make_decision(predicted_next_states, deterministic=True)
        
        return ((next_actions - predicted_next_actions) ** 2).mean() # when action is very different, loss is high
        
    def state_prediction_loss(self, indices: list, step_data: tuple):
        states, actions = step_data
        start = min(indicies)
        end   = max(indices)
        states     , actions      = states[start+0:end-1], actions[start+0:end-1]
        next_states, next_actions = states[start+1:end  ], actions[start+1:end  ]
        
        predicted_next_states = self.forward(states, actions)
        
        return ((predicted_next_states - next_states) ** 2).mean()
    
    # 
    # 
    # main training loop
    # 
    # 
    def train_with(self, env_name, agent, number_of_episodes=100, number_of_epochs=100, with_card=True, minibatch_size=settings.minibatch_size):
        env      = config.get_env(env_name)
        card     = ss.DisplayCard("multiLine", dict(train=[], test=[])) if with_card else None
        recorder = RecordKeeper(
            training_record=True,
            env_name=env_name,
            batch_size=minibatch_size,
            number_of_episodes=number_of_episodes,
            number_of_epochs=number_of_epochs,
        ).set_parent(self.recorder)
    
        # Get experience from trained agent
        episodes, all_actions, all_curr_states, all_next_states = agent.gather_experience(env, number_of_episodes)
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
        
        # 
        # timestep
        # 
        for epochs_index in range(number_of_epochs):
            # 
            # training
            # 
            self.train(True)
            train_losses = []
            indicies = range(len(train_states))
            for indicies_bundle in bundle(indicies, bundle_size=minibatch_size):
                # a tiny bundles would break some of the loss functions (because they look ahead/behind)
                if len(indicies_bundle) < 3:
                    continue
                train_losses.append(self.timestep_training_loss(indicies_bundle, tuple(train_states, train_actions)))
            train_loss = to_tensor(train_losses).mean()
            
            # 
            # testing
            # 
            self.train(False)
            test_loss = self.timestep_testing_loss(range(len(testing_data)), testing_data)
            
            print(f"    Epoch {epochs_index+1}. Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")
            recorder.push(
                epochs_index=epochs_index,
                train_loss=train_loss,
                test_loss=test_loss,
            )
            if card: card.send(dict(
                train=[epochs_index, train_loss],
                test=[epochs_index, test_loss],
            ))
        
        return recorder
    
    def generate_graphs(self):
        training_records = tuple(each for each in self.recorder.records if each.get("training_record", False))
        ss.DisplayCard("multiLine", dict(
            train=[ (each.epochs_index, each.train_loss) for each in training_records ],
            test=[ (each.epochs_index, each.test_loss) for each in training_records ],
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
        coach.recorder = RecordKeeper.load(path_to.recorder)
        return coach