
import numpy as np
import torch
import gym
import file_system_py as FS
from stable_baselines3 import SAC
from trivial_torch_tools import to_tensor, init, convert_each_arg

from info import path_to, config

class Agent(SAC):
    # 
    # load
    # 
    @classmethod
    def smart_load(cls, env_name, *, path, iterations=config.train_agent.iterations, force_retrain=config.train_agent.force_retrain):
        # skip already-trained ones
        if not force_retrain:
            if FS.exists(path+".zip"):
                print(f'''model exists: {path}.zip, loading that instead of training it''')
                return Agent.load(
                    path,
                    config.get_env(env_name),
                    device=config.device,
                )
        
        print(f'''\n\n-------------------------------------------------------''')
        print(f''' training agent from scratch: {path}.zip''')
        print(f'''-------------------------------------------------------\n\n''')
        # train and return
        agent = Agent("MlpPolicy", env_name, device=config.device, verbose=2,)
        agent.learn(iterations)
        agent.save(FS.clear_a_path_for(path_to.agent_model_for(env_name), overwrite=True))
        return agent
            
    # 
    # load
    # 
    @classmethod
    def load_default_for(cls, env_name, *, load_previous_weights=True):
        if load_previous_weights:
            return Agent.load(
                path_to.agent_model_for(env_name),
                config.get_env(env_name),
                device=config.device,
            )
        else:
            return Agent(
                "MlpPolicy",
                env_name,
                device=config.device,
                verbose=2,
            )

    # add freeze methods (for when CoachClass uses agent)
    @init.freeze_tools()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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