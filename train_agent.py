import gym

from stable_baselines3 import SAC

from info import path_to, config

def load_agent(env_name):
    if "Humanoid" in env_name:
        agent_path = path_to.file.humanoid_agent_model
    else:
        agent_path = path_to.agent_model_for(env_name)
    
    return SAC.load(
        agent_path,
        config.get_env(env_name),
        device=config.device,
    )

# 
# train
# 
if __name__ == '__main__':
    for env_name in config.default_env_names:
        SAC(
            "MlpPolicy",
            env_name,
            device=config.device,
            verbose=2
        ).learn(
            config.train_agent.iterations
        ).save(
            path_to.agent_model_for(env_name)
        )
