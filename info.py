from quik_config import find_and_load, LazyDict
from super_hash import super_hash

# 
# load the options
# 
info = find_and_load(
    "info.yaml",
    cd_to_filepath=True,
    parse_args=True,
    defaults_for_local_data=[],
)
config                = info.config         # the resulting dictionary for all the selected options
path_to               = info.path_to               # a dictionary of paths relative to the root_path
absolute_path_to      = info.absolute_path_to      # same dictionary of paths, but made absolute

# 
# get experiment name
# 
if not config.experiment_name:
    raise Exception(f'''Please give an experiment name\nex:\n    python thing.py -- experiment_name:test1\n''')

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
path_to.agent_model_for           = lambda env_name: f"{path_to.folder.agent_models}/{env_name}/{config.train_agent.model_name}"
path_to.coach_model_for           = lambda env_name: f"{path_to.folder.coach_models}/{env_name}/{config.experiment_name}/"
path_to.test_results              = lambda env_name: f"{path_to.folder.results}/{config.experiment_name}/{env_name}/test_data/"
path_to.experiment_csv_for        = lambda env_name: f"{path_to.test_results(env_name)}/experiments.csv"
path_to.experiment_visuals_folder = lambda env_name: f"{path_to.folder.results}/{config.experiment_name}/{env_name}/visuals/"

# 
# env lookup
# 
import gym
import pybullet_envs
def get_env(env_name):
    # might intercept names in the future 
    env = gym.make(env_name)
    env.name = env_name
    return env
config.get_env = get_env
# tell super_hash that pandas dataframes should be converted to csv, then hashed
super_hash.conversion_table[gym.Env] = lambda env : super_hash(env.name)

# 
# patch np.array
# 
from slick_siphon import siphon
import torch
import numpy as np
# wrap to_torch_tensor with a siphon!
# -> when the lambda returns true
# -> the function below is run INSTEAD of the original np.array()
@siphon(when=(lambda *args, **kwargs: isinstance(args[0], torch.Tensor)), is_true_for=np.array)
def custom_handler(tensor):
    # always send to cpu first (otherwise it fails)
    return tensor.cpu().detach().numpy()