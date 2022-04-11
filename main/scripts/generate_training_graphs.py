from analysis import plot_epsilon_1, plot_epsilon_2
from info import path_to, config

env_names = config.env_names

for each in env_names: plot_epsilon_1(each)
for each in env_names: plot_epsilon_2(each)