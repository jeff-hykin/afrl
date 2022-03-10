import gym

from stable_baselines3 import SAC

from info import path_to, config

def load_agent(env_name):
    return SAC.load(
        path_to.agent_model_for(env_name),
        gym.make(env_name),
        device=config.device,
    )

# 
# train
# 
if __name__ == '__main__':
    for env_name in config.env_names:
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
