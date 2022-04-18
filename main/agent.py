import itertools

import numpy as np
import torch
import gym
import file_system_py as FS
from stable_baselines3 import SAC
from trivial_torch_tools import to_tensor, init, convert_each_arg

from smart_cache import cache
from info import path_to, config
from tools import flatten, get_discounted_rewards, divide_chunks, minibatch, ft, Episode, train_test_split, TimestepSeries, to_numpy, feed_forward, bundle, average

class Agent(SAC):
    # 
    # load
    # 
    @classmethod
    def smart_load(cls, env_name, *, path, iterations=config.train_agent.iterations, force_retrain=config.train_agent.force_retrain):
        # skip already-trained ones
        if not force_retrain:
            if ".zip" not in path:
                path = f"{path}.zip"
            if FS.exists(path):
                print(f'''\n\n-----------------------------------------------------------------------------------------------------''')
                print(f''' Agent Model Exists, loading: {path}''')
                print(f'''-----------------------------------------------------------------------------------------------------\n\n''')
                agent = Agent.load(
                    path,
                    config.get_env(env_name),
                    device=config.device,
                    custom_objects = { # a hack for loading from stable baselines: https://github.com/DLR-RM/stable-baselines3/pull/336
                        "learning_rate": 0.0,
                        "lr_schedule": lambda _: 0.0,
                        "clip_range": lambda _: 0.0,
                    },
                )
                agent.path = path
                return agent
        
        print(f'''\n\n-------------------------------------------------------''')
        print(f''' Training Agent from scratch for {path}''')
        print(f'''-------------------------------------------------------\n\n''')
        # train and return
        agent = Agent("MlpPolicy", env_name, device=config.device, verbose=2,)
        agent.learn(iterations)
        agent.path = path
        agent.save(FS.clear_a_path_for(path, overwrite=True))
        return agent
            
    # add freeze methods (for when Coach uses agent)
    @init.freeze_tools()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.average_episode_reward = None
    # this is for freeze()
    def children(self):
        return [self.critic, self.actor]
    
    # 
    # predict
    # 
    def predict(self, state, **kwargs):
        """
            state: np.array().shape = (2,)
            @return: tuple(np.array().shape=(1,), None)
        """
        if isinstance(state, torch.Tensor):
            state = state.detach()
        return super().predict(state, **kwargs)
    
    # 
    # Actor Policy
    # 
    def make_decision(self, state_batch, deterministic=True):
        state_batch = to_tensor(state_batch).to(self.device)
        # must use forward instead of predict to preserve tensor tracking
        action_batch = self.actor.forward(state_batch, deterministic=deterministic)
        return action_batch
    
    # 
    # Q function
    # 
    @convert_each_arg.to_batched_tensor(number_of_dimensions=2)
    def value_of(self, state, action):
        """
            batched:
                state.shape = torch.Size([32, 2])
                action.shape = torch.Size([32, 1])
        """
        action = to_tensor(action).to(self.device)
        state = to_tensor(state).to(self.device)
        result = self.critic_target(state, action)
        q = torch.cat(result, dim=1)
        q, _ = torch.min(q, dim=1, keepdim=True)
        return q
    
    # alternative Q functions found in the code:
        # def Q(agent, state: np.ndarray, action: np.ndarray):
        #     if torch.is_tensor(action):
        #         action = torch.unsqueeze(action, 0).to(config.device)
        #     else:
        #         action = ft([action])
        #     q = torch.cat(agent.critic_target(ft([state]), action), dim=1)
        #     q, _ = torch.min(q, dim=1, keepdim=True)
        #     return q

        # def Q(agent, state: np.ndarray, action: np.ndarray):
        #     state = ft([state]).to(agent.device)
        #     action = ft([action]).to(agent.device)
        #     with torch.no_grad():
        #         q = torch.cat(agent.critic_target(state, action), dim=1)
        #     q, _ = torch.min(q, dim=1, keepdim=True)
        #     return q.item()
    
    @cache(depends_on=[config.train_agent])
    def gather_experience(self, env, number_of_episodes):
        episodes = [None]*number_of_episodes
        reward_per_episode = []
        reward_per_timestep = []
        all_actions, all_curr_states = [], []
        print(f'''Starting Experience Recording''')
        for episode_index in range(number_of_episodes):
            episode = Episode()
            
            state = env.reset()
            episode.states.append(state)
            episode.reward_total = 0
            
            done = False
            while not done:
                action, _ = self.predict(state, deterministic=True)
                action = np.random.multivariate_normal(action, 0 * np.identity(len(action))) 
                # QUESTION: why sample from multivariate_normal?
                # Answer: because of state-space exploration and robustness
                # The deterministic=True is just so that we have exact control over the kind of randomness
                # Good agents will stick to good paths, so to get a more-representative explortation of the state space randomness is used
                episode.actions.append(action)
                state, reward, done, info = env.step(action)
                reward_per_timestep.append(reward)
                episode.states.append(state)
                episode.reward_total += reward
            
            reward_per_episode.append(episode.reward_total)
            print(f"    Episode: {episode_index}, Reward: {episode.reward_total:.3f}, Average Reward: {average(reward_per_episode):.3f}")
            episodes[episode_index] = episode
            all_curr_states += episode.curr_states
            all_actions     += episode.actions
        
        self.average_episode_reward = average(reward_per_episode)
        self.average_timestep_reward = average(reward_per_timestep)
        print(f'''''')
        print(f'''    Max Episode Reward: {max(reward_per_episode)}''')
        print(f'''    Min Episode Reward: {min(reward_per_episode)}''')
        print(f'''    Max Timestep Reward: {max(reward_per_timestep)}''')
        print(f'''    Min Timestep Reward: {min(reward_per_timestep)}''')
        return episodes, all_actions, all_curr_states
    
    def __super_hash__(self):
        return self.path
