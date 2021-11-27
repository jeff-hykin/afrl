import gym

from stable_baselines3 import SAC

env_name = "MountainCarContinuous-v0"
SAC("MlpPolicy", env_name, verbose=2).learn(100_000).save(f'data/{env_name}')
