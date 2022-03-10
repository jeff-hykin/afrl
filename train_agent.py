import gym

from stable_baselines3 import SAC

from info import path_to, config

for env_name in config.env_names:
    SAC(
        "MlpPolicy",
        env_name,
        device=config.device,
        verbose=2
    ).learn(
        config.train_agent.iterations
    ).save(
        f'{path_to.folders.agent_models}{env_name}'
    )
