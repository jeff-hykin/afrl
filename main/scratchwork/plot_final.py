import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybullet_envs
import scipy.stats as st
import seaborn as sns
import silver_spectacle as ss
from rigorous_recorder import RecordKeeper
from tools import average
from random import random, sample, choices, shuffle
import json
from os.path import join

import file_system_py as FS
from super_map import LazyDict
from rigorous_recorder import Recorder

from info import config, path_to
from tools import average

def shuffled(thing):
    a = list(thing)
    shuffle(a)
    return a


# For each #value create a plot for each env of the theory bound (function of epsilon) and the emperical bound

# f"{path_to.folder.results}/final_1/{config.env_name}/{config.predictor_settings.acceptable_performance_loss}/{config.predictor_settings.method}",

source = f"{path_to.folder.results}/final_1"


# final_1/lunar/ppac/0.80/
# final_1/lunar/ppac/0.90/
# final_1/lunar/ppac/0.95/
# final_1/lunar/ppac/0.98/
# final_1/lunar/ppac/0.99/
# final_1/lunar/n_step
# final_1/lunar/optimal
# final_1/lunar/random



plot_data_per_env = LazyDict().setdefault(
    lambda each_env: LazyDict(
        rewards=LazyDict(), # <- keys are methods
        failure_points=LazyDict(),
    )
)
for each_env_path in FS.list_folder_paths_in(source):
    env_name = FS.basename(each_env_path)
    plot_data_per_method = plot_data_per_env[env_name]
    
    for each_method_path in FS.list_folder_paths_in(each_env_path):
        method = FS.basename(each_method_path)
        
        for each_performance_level_path in FS.list_folder_paths_in(each_method_path):
            each_performance_level = float(FS.basename(each_performance_level_path))
            
            # load simple data
            # simple_data = None
            # with open(f"{each_performance_level}/simple_data.json", 'r') as in_file:
            #     simple_data = json.load(in_file)
            # simple_data = LazyDict(simple_data)
            # performance_level = simple_data.acceptable_performance_level
            
            # load detailed data
            recorder = Recorder.load_from(f"{each_performance_level}/serial_data/recorder.pickle").full
            average_plan_length = average(recorder.frame["failure_point_average"])
            average_episode_reward = average(recorder.frame["discounted_reward_sum"])
            
            plot_data_per_method.rewards[]
            
            # save to plottable data
            plot_data.rewards[performance_level] = average_plan_length
            plot_data.failure_points[performance_level] = average_plan_length
    
    

Recorder.load_from('results/final_1_80%/HopperBulletEnv-v0/serial_data/rewards_per_episode_per_timestep.pickle')

episode_index
timestep_count
forecast_average
alt_forecast_average
normalized_reward_average
discounted_reward
failure_point

agent_data = dict(
    agent1= [
        [ 0,  -59.744 ],
        [ 1,  6.558 ],
        [ 2,  23.390 ],
        [ 3,  -14.323 ],
        [ 4,  -7.273 ],
        [ 5,  -36.950 ],
        [ 6,  -29.815 ],
        [ 7,  -8.679 ],
        [ 8,  17.649 ],
        [ 9,  -19.233 ],
        [ 10, -5.321 ],
        [ 11, -33.234 ],
        [ 12, 12.891 ],
        [ 13, -24.156 ],
        [ 14, -11.600 ],
        [ 15, -23.465 ],
        [ 16, -20.254 ],
        [ 17, 2.922 ],
        [ 18, -6.967 ],
        [ 19, -8.541 ],
    ],
    agent2= [
        [ 0,  -35.654 ],
        [ 1,  220.118 ],
        [ 2,  189.385 ],
        [ 3,  23.082 ],
        [ 4,  212.490 ],
        [ 5,  -249.450 ],
        [ 6,  -19.394 ],
        [ 7,  214.021 ],
        [ 8,  231.051 ],
        [ 9,  253.862 ],
        [ 10, -34.156 ],
        [ 11, -9.270 ],
        [ 12, -17.916 ],
        [ 13, -314.552 ],
        [ 14, -35.948 ],
        [ 15, -29.119 ],
        [ 16, -54.415 ],
        [ 17, 261.187 ],
        [ 18, 204.776 ],
        [ 19, -176.926 ],
    ],
    agent3= [
        [ 0,  -8.560 ],
        [ 1,  -10.912 ],
        [ 2,  229.472 ],
        [ 3,  124.006 ],
        [ 4,  135.132 ],
        [ 5,  243.481 ],
        [ 6,  -19.292 ],
        [ 7,  -36.580 ],
        [ 8,  -129.515 ],
        [ 9,  -20.519 ],
        [ 10, 114.468 ],
        [ 11, 214.642 ],
        [ 12, 241.050 ],
        [ 13, -23.469 ],
        [ 14, -92.102 ],
        [ 15, 225.902 ],
        [ 16, -32.678 ],
        [ 17, 208.497 ],
        [ 18, 217.153 ],
        [ 19, 202.226 ],
    ],
    agent4= [
        [ 0,  -12.221 ],
        [ 1,  -30.900 ],
        [ 2,  -29.749 ],
        [ 3,  -2.754 ],
        [ 4,  -29.175 ],
        [ 5,  2.851 ],
        [ 6,  -51.666 ],
        [ 7,  -42.954 ],
        [ 8,  -21.758 ],
        [ 9,  -12.622 ],
        [ 10, 9.328 ],
        [ 11, -15.155 ],
        [ 12, -8.899 ],
        [ 13, -44.338 ],
        [ 14, 8.511 ],
        [ 15, 8.498 ],
        [ 16, -17.201 ],
        [ 17, 1.048 ],
        [ 18, -37.873 ],
        [ 19, -19.898 ],
    ],
    agent5= [
        [ 0,  262.856 ],
        [ 1,  227.429 ],
        [ 2,  216.260 ],
        [ 3,  252.914 ],
        [ 4,  246.682 ],
        [ 5,  238.758 ],
        [ 6,  270.917 ],
        [ 7,  265.426 ],
        [ 8,  290.499 ],
        [ 9,  242.949 ],
        [ 10, 275.475 ],
        [ 11, 266.616 ],
        [ 12, 235.471 ],
        [ 13, 249.150 ],
        [ 14, 272.340 ],
        [ 15, 5.625 ],
        [ 16, 2.309 ],
        [ 17, 292.349 ],
        [ 18, 287.616 ],
        [ 19, 282.204 ],
    ],
    agent6=[
        [ 0, 198.094 ],
        [ 1, 210.719 ],
        [ 2, 257.565 ],
        [ 3, 208.475 ],
        [ 4, 196.943 ],
        [ 5, -259.806 ],
        [ 6, 236.045 ],
        [ 7, 6.005 ],
        [ 8, 190.025 ],
        [ 9, 199.222 ],
        [ 10, 270.852 ],
        [ 11, 185.579 ],
        [ 12, 258.002 ],
        [ 13, 205.693 ],
        [ 14, 31.881 ],
        [ 15, 244.119 ],
        [ 16, 234.227 ],
        [ 17, 260.150 ],
        [ 18, 254.176 ],
        [ 19, 214.015 ],
    ],
    agent7=[
        [ 0, 283.121],
        [ 1, 241.191],
        [ 2, 273.269],
        [ 3, 268.009],
        [ 4, 38.360],
        [ 5, 260.585],
        [ 6, 268.131],
        [ 7, 269.923],
        [ 8, -98.192],
        [ 9, 280.412],
        [ 10, 268.772],
        [ 11, 234.468],
        [ 12, 22.191],
        [ 13, 184.948],
        [ 14, -25.474],
        [ 15, 246.749],
        [ 16, -9.343],
        [ 17, 261.288],
        [ 18, -22.920],
        [ 19, 274.421],
    ],
    agent8=[
        [ 0, 243.798],
        [ 1, 236.465],
        [ 2, 173.795],
        [ 3, 18.985],
        [ 4, 159.833],
        [ 5, 131.485],
        [ 6, -315.151],
        [ 7, 250.094],
        [ 8, 243.633],
        [ 9, 198.805],
        [ 10, 188.470],
        [ 11, 252.859],
        [ 12, 189.699],
        [ 13, 176.762],
        [ 14, 176.090],
        [ 15, 199.793],
        [ 16, 144.329],
        [ 17, -190.681],
        [ 18, 223.339],
        [ 19, 207.648],
    ],
    agent9=[
        [ 0, -9.511],
        [ 1, 157.125],
        [ 2, -172.318],
        [ 3, -107.149],
        [ 4, 232.383],
        [ 5, -51.048],
        [ 6, 186.500],
        [ 7, 133.603],
        [ 8, 219.569],
        [ 9, 194.224],
        [ 10, 265.178],
        [ 11, 228.879],
        [ 12, 154.790],
        [ 13, -277.035],
        [ 14, 203.311],
        [ 15, 170.673],
        [ 16, 249.733],
        [ 17, 9.263],
        [ 18, 15.511],
        [ 19, 292.199],
    ],
    agent10=[
        [ 0, -83.537],
        [ 1, -43.608],
        [ 2, -82.160],
        [ 3, -94.292],
        [ 4, -90.012],
        [ 5, -133.568],
        [ 6, -38.642],
        [ 7, -22.207],
        [ 8, -67.589],
        [ 9, -53.407],
        [ 10, -65.722],
        [ 11, -52.438],
        [ 12, -66.102],
        [ 13, -50.114],
        [ 14, -41.827],
        [ 15, -104.453],
        [ 16, -56.099],
        [ 17, -20.305],
        [ 18, -82.338],
        [ 19, -31.986],
    ],
)
averages = tuple(
    average(each_element[1] for each_element in each_agent_scores)
        for each_agent_scores in agent_data.values()
)


consistency = dict(
    agent1=[ [ 1, -10.220284512581983 ], [ 2, -12.30725 ], ],
    agent2=[ [ 1, -19.510836942999056 ], [ 2, 89.1200482846428 ], ],
    agent3=[ [ 1, -22.735617832924518 ], [ 2, -17.346413752891536 ], ],
    agent4=[ [ 1, 246.22719994526506 ], [ 2, 234.19222777728007 ], ],
    agent5=[ [ 1, 173.9950409026723 ], [ 2, 180.09900689766872 ], ],
    agent6=[ [ 1, 196.51549273727676 ], [ 2, 175.99533212924376 ], ],
    agent7=[ [ 1, 190.14854944419437 ], [ 2, 145.50251724411746 ], ],
    agent8=[ [ 1, -43.79013598056463 ], [ 2, 104.79396467624586 ], ],
    agent9=[ [ 1, -54.04997149052516 ], [ 2, -64.02022260411694 ], ],
    agent10=[ [ 1, 2.9295989148901582 ], [ 2, 70.18083292489688 ], ],
)

# ss.DisplayCard("multiLine", agent_data)
# ss.DisplayCard("quickScatter", shuffled(averages))
ss.DisplayCard("multiLine", consistency)