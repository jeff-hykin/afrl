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
    # loss_api
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
            loss_api=settings.loss_api,
            number_of_episodes=settings.number_of_episodes,
            number_of_epochs=settings.number_of_epochs,
            with_card=settings.with_card,
            minibatch_size=settings.minibatch_size,
        )
        
        # 
        # save
        # 
        coach.save(path)
        if settings.with_card: coach.generate_training_card()
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
    
    def timestep_testing_loss(self, indices: list, step_data: tuple):
        self.eval() # testing mode
        loss_function = getattr(self, self.which_loss)
        loss = loss_function(indices, step_data)
        return to_pure(loss)
    
    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="device")
    def batch_testing_loss(self, state_batch: torch.Tensor, action_batch: torch.Tensor, next_state_batch: torch.Tensor):
        self.eval() # testing mode
        loss_function = getattr(self, self.which_loss)
        loss = loss_function(state_batch, action_batch, next_state_batch)
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
    
    @convert_each_arg.to_tensor()
    @convert_each_arg.to_device(device_attribute="device")
    def batch_training_loss(self, state_batch: torch.Tensor, action_batch: torch.Tensor, next_state_batch: torch.Tensor):
        loss_function = getattr(self, self.which_loss)
        self.agent.freeze()
        loss = loss_function(state_batch, action_batch, next_state_batch)
        
        # Optimize the self model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.agent.unfreeze()
        return loss
    
    # 
    # Loss function options (loss_api=timestep)
    # 
    def consistent_coach_loss(self, indices: list, step_data: tuple):
        losses = []
        scale_value_prediction = self.settings.consistent_coach_loss.scale_value_prediction
        for index in indices:
            # need multiple to penalize the future
            if index < 1:
                continue
            
            state1, action1, state2 = step_data[index]
            value_prediction_loss = self.value_prediction_loss([state1], [action1], [state2]) * scale_value_prediction
            losses.append(value_prediction_loss)
            
            state1, action1, state2 = step_data[index-1]
            state2, action2, state3 = step_data[index-0]
            
            # TODO: check how this is going to effect vectorization/batches
            once_predicted_state2  = self.forward(state1               , action1)
            twice_predicted_state3 = self.forward(once_predicted_state2, action2)
            once_predicted_state3  = self.forward(state2               , action2)
            
            future_loss = ((once_predicted_state3 - twice_predicted_state3)**2).mean()
            losses.append(future_loss)
        
        return torch.stack(losses).mean()
    
    def timestep_loss(self, timesteps):
        predictions = self.create_forcast(timesteps.steps[0].prev_state, timesteps.steps[0].action, len(timesteps.steps))
        losses = []
        for (predicted_next_state, predicted_action), real in zip(predictions, timesteps.steps):
            predicted_value = self.agent.value_of(real.next_state, predicted_action)
            actual_value    = self.agent.value_of(real.next_state, real.action)
            losses.append(actual_value - predicted_value)
        return torch.stack(losses).mean()
    
    # 
    # Loss function options (loss_api=batched)
    # 
    @convert_each_arg.to_tensor()
    def value_prediction_loss(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        predicted_next_state   = self.forward(state, action)
        
        predicted_next_action = self.agent.make_decision(predicted_next_state, deterministic=True)
        predicted_next_value  = self.agent.value_of(next_state, predicted_next_action)
        best_next_action = self.agent.make_decision(next_state, deterministic=True)
        best_next_value  = self.agent.value_of(next_state, best_next_action)
        
        return (best_next_value - predicted_next_value).mean() # when predicted_next_value is high, loss is low (negative)
    
    def action_prediction_loss(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        predicted_next_state   = self.forward(state, action)
        
        predicted_next_action = self.agent.make_decision(predicted_next_state, deterministic=True)
        best_next_action = self.agent.make_decision(next_state, deterministic=True)
        
        return ((best_next_action - predicted_next_action) ** 2).mean() # when action is very different, loss is high
        
    def state_prediction_loss(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        predicted_next_state = self.forward(state, action)
        
        actual = predicted_next_state
        expected = next_state
        
        return ((actual - expected) ** 2).mean()

    def train_with(self, env_name, agent, loss_api, number_of_episodes=100, number_of_epochs=100, with_card=True, minibatch_size=settings.minibatch_size):
        env      = config.get_env(env_name)
        card     = ss.DisplayCard("multiLine", dict(train=[], test=[])) if with_card else None
        recorder = RecordKeeper(
            training_record=True,
            env_name=env_name,
            batch_size=minibatch_size,
            number_of_episodes=number_of_episodes,
            number_of_epochs=number_of_epochs,
            loss_api=loss_api,
        ).set_parent(self.recorder)
    
        # Get experience from trained agent
        episodes, all_actions, all_curr_states, all_next_states = agent.gather_experience(env, number_of_episodes)
        print(f'''Starting Train/Test''')
        states      = to_tensor(all_curr_states)
        actions     = to_tensor(all_actions)
        next_states = to_tensor(all_next_states)

        (
            (train_states     , test_states     ),
            (train_actions    , test_actions    ),
            (train_next_states, test_next_states),
        ) = train_test_split(
            states,
            actions,
            next_states,
            split_proportion=settings.train_test_split,
        )
        
        # 
        # timestep
        # 
        if loss_api == "timestep":
            
            training_data = tuple(zip(train_states, train_actions, train_next_states))
            testing_data  = tuple(zip(test_states , test_actions , test_next_states ))
            
            for epochs_index in range(number_of_epochs):
                # 
                # training
                # 
                self.train(True)
                train_losses = []
                for indicies in bundle(range(len(training_data)), bundle_size=minibatch_size):
                    # a size-of-one bundle would break one of the loss functions
                    if len(indicies) < 2:
                        continue
                    train_losses.append(self.timestep_training_loss(indicies, training_data))
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
        # 
        # batched
        # 
        elif loss_api == "batched":
            for epochs_index in range(number_of_epochs):
                # 
                # training
                # 
                self.train(True)
                train_losses = []
                for state_batch, action_batch, next_state_batch in minibatch(minibatch_size, train_states, train_actions, train_next_states):
                    train_losses.append(self.batch_training_loss(state_batch, action_batch, next_state_batch))
                train_loss = to_tensor(train_losses).mean()
                
                # 
                # testing
                # 
                self.train(False)
                test_loss = self.batch_testing_loss(test_states, test_actions, test_next_states)
                
                print(f"    Epoch {epochs_index+1}. Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                recorder.push(
                    epochs_index=epochs_index,
                    train_loss=train_loss,
                    test_loss=test_loss,
                )
                if card: card.send(dict(
                    train=[epochs_index, train_loss],
                    test=[epochs_index, test_loss],
                ))
        else:
            raise Exception(f'''unknown loss_api given to train coach:\n    was given: {loss_api}\n    valid values: "batched", "timestep" ''')
        
        return recorder
    
    def generate_training_card(self):
        training_records = tuple(each for each in self.recorder.records if each.get("training_record", False))
        ss.DisplayCard("multiLine", dict(
            train=[ (each.epochs_index, each.train_loss) for each in training_records ],
            test=[ (each.epochs_index, each.test_loss) for each in training_records ],
        ))
    
    @classmethod
    def internal_paths(cls, path):
        return LazyDict(
            model      = f"{path}/model.pt",
            attributes = f"{path}/attributes.pickle",
            recorder   = f"{path}/recorder.pickle",
        )
    
    def save(self, path=None):
        path_to = Coach.internal_paths(path or self.path)
        
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