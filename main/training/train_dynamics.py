import os
from posixpath import basename

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

from info import path_to, config
from main.training.train_agent import Agent
from main.tools import flatten, get_discounted_rewards, divide_chunks, minibatch, ft, Episode, train_test_split, TimestepSeries, to_numpy, feed_forward, bundle

settings = config.train_dynamics
# contains:
    # learning_rate
    # hidden_sizes
    # number_of_episodes
    # number_of_epochs
    # minibatch_size
    # loss_api
    # etc

# State transition dynamics model
class DynamicsModel(nn.Module):
    """
    The model of how the world works
        (state) => (next_state)
    """
    
    # 
    # load
    # 
    @classmethod
    def smart_load(cls,
        env_name,
        *,
        path,
        agent,
        force_train=settings.force_retrain,
    ):
        env = config.get_env(env_name)
        dynamics = DynamicsModel(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            settings=settings.merge(settings.env_overrides.get(env_name, {}))
            agent=agent,
            path=path,
            device=config.device,
        )
        # load if exists
        if not force_retrain:
            if FS.exists(path):
                dynamics.load_state_dict(torch.load(path))
                return dynamics
        
        # 
        # train
        # 
        recorder = dynamics.train(
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
        self.save(path)
    
    # init
    @init.save_and_load_methods(model_attributes=["model"], basic_attributes=[ "hidden_sizes", "learning_rate", "obs_dim", "act_dim"])
    def __init__(self, obs_dim, act_dim, settings, device, agent, path, **kwargs):
        super().__init__()
        self.settings      = settings
        self.learning_rate = self.settings.learning_rate
        self.which_loss    = self.settings.loss_function
        self.hidden_sizes  = self.settings.hidden_sizes
        self.device        = device
        self.obs_dim       = obs_dim
        self.act_dim       = act_dim
        self.agent         = agent
        self.path          = path
        self.recorder      = RecordKeeper(
            experiment_name=config.experiment_name,
            model="coach",
        )
        self.model = feed_forward(layer_sizes=[obs_dim + act_dim, *hidden_sizes, obs_dim], activation=nn.ReLU).to(self.device)
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
    # Loss function options
    # 
    def consistent_coach_loss(self, indices: list, step_data: tuple):
        losses = []
        for index in indices:
            # need multiple to penalize the future
            if index < 1:
                continue
            
            state1, action1, state2 = step_data[index]
            value_prediction_loss = self.value_prediction_loss([state1], [action1], [state2])
            losses.append(value_prediction_loss)
            
            state1, action1, state2 = step_data[index-1]
            state2, action2, state3 = step_data[index-0]
            
            # TODO: check how this is going to effect vectorization/batches
            once_predicted_state2  = self.forward(state1               , action1)
            twice_predicted_state3 = self.forward(once_predicted_state2, action2)
            once_predicted_state3  = self.forward(state2               , action2)
            
            future_loss = ((once_predicted_state3 - twice_predicted_state3)**2).mean()
            losses.append(future_loss)
        
        # FIXME: there is a problem here, which is that these losses may be on totally different scales
        #       some kind of coefficent (ideal self-tuning coefficient) is needed here
        return torch.stack(losses).mean()
    
    def timestep_loss(self, timesteps):
        predictions = self.create_forcast(timesteps.steps[0].prev_state, timesteps.steps[0].action, len(timesteps.steps))
        losses = []
        for (predicted_next_state, predicted_action), real in zip(predictions, timesteps.steps):
            predicted_value = self.agent.value_of(real.next_state, predicted_action)
            actual_value    = self.agent.value_of(real.next_state, real.action)
            losses.append(actual_value - predicted_value)
        return torch.stack(losses).mean()
    
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

    def experience(self, env, agent, number_of_episodes):
        episodes = [None]*number_of_episodes
        all_actions, all_curr_states, all_next_states = [], [], []
        print(f'''Starting Experience Recording''')
        for episode_index in range(number_of_episodes):
            episode = Episode()
            
            state = env.reset()
            episode.states.append(state)
            episode.reward_total = 0
            
            done = False
            while not done:
                action, _ = agent.predict(state, deterministic=True)  # False?
                action = np.random.multivariate_normal(action, 0 * np.identity(len(action))) # QUESTION: why sample from multivariate_normal?
                episode.actions.append(action)
                state, reward, done, info = env.step(action)
                episode.states.append(state)
                episode.reward_total += reward
            
            print(f"    Episode: {episode_index}, Reward: {episode.reward_total:.3f}")
            episodes[episode_index] = episode
            all_curr_states += episode.curr_states
            all_actions     += episode.actions
            all_next_states += episode.next_states

        return episodes, all_actions, all_curr_states, all_next_states

    def train(self, env_name, agent, loss_api, number_of_episodes=100, number_of_epochs=100, with_card=True, minibatch_size=settings.minibatch_size):
        env      = config.get_env(env_name)
        card     = ss.DisplayCard("multiLine", dict(train=[], test=[])) if with_card else None
        recorder = RecordKeeper(
            env_name=env_name,
            batch_size=minibatch_size,
            number_of_episodes=number_of_episodes,
            number_of_epochs=number_of_epochs,
            loss_api=loss_api,
        ).set_parent(self.recorder)
    
        # Get experience from trained agent
        episodes, all_actions, all_curr_states, all_next_states = self.experience(env, agent, number_of_episodes)
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
                
                print(f"    Epoch {epochs_index+1}. Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                recorder.push(
                    epochs_index=epochs_index,
                    train_loss=train_loss,
                    test_loss=test_loss,
                )
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
        else:
            raise Exception(f'''unknown loss_api given to train dynamics:\n    was given: {loss_api}\n    valid values: "batched", "timestep" ''')
        
        return recorder
    
    def save(self, path=None):
        path = path or self.path
        torch.save(self.state_dict(), FS.clear_a_path_for(path, overwrite=True))
        self.recorder.save_to(FS.clear_a_path_for(path+".records", overwrite=True))