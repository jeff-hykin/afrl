from quik_config import find_and_load, LazyDict

# 
# load the options
# 
info = find_and_load(
    "info.yaml",
    cd_to_filepath=True,
    parse_args=True,
    defaults_for_local_data=[ "ENVS=BASIC", ],
)
config                = info.config         # the resulting dictionary for all the selected options
path_to               = info.path_to               # a dictionary of paths relative to the root_path
absolute_path_to      = info.absolute_path_to      # same dictionary of paths, but made absolute

# 
# set torch device
# 
import torch
config.device = torch.device('cpu')
if not config.force_cpu:
    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 
# create any missing folders
# 
for each_name in path_to.folder:
    each_path = path_to.folder[each_name]
    import os
    os.makedirs(each_path, exist_ok=True)

# 
# functional paths
# 
path_to.agent_model_for    = lambda env_name: f"{path_to.folder.agent_models}/{env_name}"
path_to.dynamics_model_for = lambda env_name: f"{path_to.folder.dynamics_models}/{env_name}.pt"
path_to.experiment_csv_for = lambda env_name: f"{path_to.folder.results}/{env_name}/experiments_pp.csv"

# 
# env lookup
# 
import gym
import pybullet_envs
config.get_env = lambda env_name: gym.make(env_name) # might intercept this in the future 