from quik_config import find_and_load, LazyDict

# 
# load the options
# 
info = find_and_load("info.yaml", default_options=[])
config                = info.config         # the resulting dictionary for all the selected options
path_to               = info.path_to               # a dictionary of paths relative to the root_path
absolute_path_to      = info.absolute_path_to      # same dictionary of paths, but made absolute
project               = info.project               # the dictionary to everything inside (project)
root_path             = info.root_path             # parent folder of the .yaml file
configuration_choices = info.configuration_choices # the dictionary of the local config-choices files
configuration_options = info.configuration_options # the dictionary of all possible options
as_dict               = info.as_dict               # the dictionary to the whole file (info.yaml)

# 
# set torch device
# 
if config.has_gpu:
    import torch
    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 
# create any missing folders
# 
for each_name in path_to.folders:
    each_path = path_to.folders[each_name]
    import os
    os.makedirs(each_path, exist_ok=True)

# 
# functional paths
# 
path_to.agent_model_for    = lambda env_name: f"{path_to.folders.agent_models}/{env_name}"
path_to.dynamics_model_for =lambda env_name: f"{path_to.folders.dynamics_models}/{env_name}.pt"