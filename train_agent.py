import gym

from stable_baselines3 import SAC

for env_name in ["LunarLanderContinuous-v2", "MountainCarContinuous-v0"]:
    SAC("MlpPolicy", env_name, verbose=2).learn(100_000).save(f'data/{env_name}')
