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
from tools import average, multi_plot

def shuffled(thing):
    a = list(thing)
    shuffle(a)
    return a


# For each #value create a plot for each env of the theory bound (function of epsilon) and the emperical bound

source = f"{path_to.folder.results}/final_2"
for each_env_path in FS.list_folder_paths_in(source):
    env_name = FS.basename(each_env_path)
    
    # load simple data
    simple_data = None
    with open(f"{each_env_path}/simple_data.json", 'r') as in_file:
        simple_data = json.load(in_file)
    simple_data = LazyDict(simple_data)
    
    plot_data = LazyDict(simple_data.plot)
    
    # 
    # reward plot
    # 
    ss.DisplayCard("multiLine", dict(
        optimal=plot_data.optimal_reward_points,
        random=plot_data.random_reward_points,
        theory=plot_data.theory_reward_points,
        ppac=plot_data.ppac_reward_points,
        n_step_horizon=plot_data.n_step_horizon_reward_points,
        n_step_planlen=plot_data.n_step_planlen_reward_points,
    ))
    # 
    # forcast plot
    # 
    ss.DisplayCard("multiLine", dict(
        ppac=plot_data.ppac_plan_length_points,
        n_step_horizon=plot_data.n_step_horizon_plan_length_points,
        n_step_planlen=plot_data.n_step_planlen_plan_length_points,
    ))
    
    multi_plot(
        dict(
            optimal=plot_data.optimal_reward_points,
            random=plot_data.random_reward_points,
            theory=plot_data.theory_reward_points,
            ppac=plot_data.ppac_reward_points,
            n_step_horizon=plot_data.n_step_horizon_reward_points,
            n_step_planlen=plot_data.n_step_planlen_reward_points,
        ),
        vertical_label="reward",
        horizonal_label="acceptance level",
        title=None,
        color_key=dict(
            optimal='#83ecc9',
            ppac='#89ddff',
            theory='#e57eb3',
            n_step_planlen='#fec355',
            n_step_horizon='#f07178',
            random='#c7cbcd',
        )
    )


        # "#83ecc9",
        # "#89ddff",
        # "#82aaff",
        # "#c792ea",
        # "#e57eb3",
        # "#fec355",
        # "#f07178",
        # "#f78c6c",
        # "#c3e88d",
        # black          : &black          '#000000'
        # white          : &white          '#ffffff'
        # light_gray     : &light_gray     '#c7cbcd'
        # gray           : &gray           '#546e7a'
        # rust           : &rust           '#c17e70'
        # orange         : &orange         '#f78c6c'
        # yellow         : &yellow         '#fec355'
        # bananna_yellow : &bananna_yellow '#ddd790'
        # lime           : &lime           '#c3e88d'
        # green          : &green          '#4ec9b0'
        # bold_green     : &bold_green     '#4ec9b0d0'
        # vibrant_green  : &vibrant_green  '#04d895'
        # dim_green      : &dim_green      '#80cbc4' # new dim green: #80CBAB
        # dark_slate     : &dark_slate     '#3f848d'
        # light_slate    : &light_slate    '#64bac5'
        # light_blue     : &light_blue     '#89ddff'
        # blue           : &blue           '#82aaff'
        # electric_blue  : &electric_blue  '#00aeffe7'
        # light_purple   : &light_purple   '#c792ea'
        # pink           : &pink           '#e57eb3'
        # red            : &red            '#ff5572'
        # soft_red       : &soft_red       '#f07178'