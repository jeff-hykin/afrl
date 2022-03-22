
import numpy as np
import torch
import gym
from stable_baselines3 import SAC
from trivial_torch_tools import to_tensor, init, convert_each_arg

from info import path_to, config

class Agent(SAC):
    @init.add_frozen_methods()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def children(self):
        return [self.critic, self.actor]
    
    def predict(self, state, **kwargs):
        """
            state: np.array().shape = (2,)
            @return: tuple(np.array().shape=(1,), None)
        """
        if isinstance(state, torch.Tensor):
            state = state.detach()
        return super().predict(state, **kwargs)
    
    def make_decision(self, state, deterministic=True):
        state = to_tensor(state).to(self.device)
        # must use forward instead of predict to preserve tensor tracking
        actions = self.actor.forward(state, deterministic=deterministic)
        return actions
    
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
    
def load_agent(env_name):
    if "Humanoid" in env_name:
        agent_path = path_to.file.humanoid_agent_model
    else:
        agent_path = path_to.agent_model_for(env_name)
    
    return Agent.load(
        agent_path,
        config.get_env(env_name),
        device=config.device,
    )

# 
# train
# 
if __name__ == '__main__':
    for env_name in config.env_names:
        Agent(
            "MlpPolicy",
            env_name,
            device=config.device,
            verbose=2
        ).learn(
            config.train_agent.iterations
        ).save(
            path_to.agent_model_for(env_name)
        )
