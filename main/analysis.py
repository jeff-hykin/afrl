from statsmodels.stats.weightstats import ttest_ind
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybullet_envs
import scipy.stats as st
import seaborn as sns

from info import path_to, config
import file_system_py as FS

env_names = config.env_names
settings = config.gym_env_settings

sns.set_theme(style="whitegrid")

def confidence_interval(xs):
    return st.t.interval(0.95, len(xs)-1, loc=np.mean(xs), scale=st.sem(xs))

def plot_epsilon_1(env_name):
    plt.figure(figsize=(5.5, 5.5))
    
    if not FS.exists(path_to.experiment_csv_for(env_name)):
        print(f"no data found for: {env_name}: {path_to.experiment_csv_for(env_name)}")
        return
    
    ax = plt.subplot(1,1,1)
    afrl = pd.read_csv(path_to.experiment_csv_for(env_name)).rename(columns={'discounted_rewards': 'V'})
    vmax, vmin = settings[env_name]['max_score'], settings[env_name]['min_score']
    vmax = afrl.groupby('epsilon').V.mean().values[0]
    vrange = vmax - vmin
    afrl['opt'] = (afrl.V - vmin) / vrange
    afrl['eps'] = afrl.epsilon / vrange
    base_mean = 1
    af_means = afrl.groupby('eps').opt.mean()
    sd = afrl.groupby('eps').opt.std().values
    epsilon_low, epsilon_hgh = afrl.eps.min(), afrl.eps.max()
    epsilon = np.linspace(0, epsilon_hgh, 2)
    gamma = settings[env_name].agent_discount_factor
    subopt = epsilon / (1 - gamma)
    plt.plot(epsilon, base_mean-subopt, color='orange', label='Perform. Bounds')
    af_means.plot(label='V^{PAC}(s_0)', marker='o', ax=ax, color='steelblue')
    plt.fill_between(af_means.index, af_means-sd, af_means+sd, alpha=0.2, color='steelblue')
    plt.hlines(base_mean, epsilon_low, epsilon_hgh, color='green', label='V^pi(s_0)')

    plt.plot([epsilon_low, epsilon_hgh], [0,0], color='red', label='V^{rand}(s_0)')

    plt.xlabel('Epsilon Coefficient')

    plt.ylim(-.1, 1.1)

    plt.legend(loc='lower left', fontsize="x-small")
    plt.title(env_name)
    
    plt.tight_layout()
    file_path = path_to.experiment_visuals_folder(env_name) + "epsilon_1"
    plt.savefig(FS.clear_a_path_for(file_path, overwrite=True))

def plot_epsilon_2(env_name):
    plt.figure(figsize=(5.5, 5.5))
    
    afrl = pd.read_csv(path_to.experiment_csv_for(env_name)).rename(columns={'discounted_rewards': 'V', 'Unnamed: 0': 'episode'})
    vmax, vmin = settings[env_name]['max_score'], settings[env_name]['min_score']
    vrange = vmax - vmin
    afrl['opt'] = (afrl.V - vmin) / vrange
    afrl['eps'] = afrl.epsilon / vrange

    f_means = afrl.groupby('eps').forecast.mean() + 1
    standard_deviations = afrl.groupby(['eps', 'episode']).forecast.mean().unstack().std(1)

    f_low, f_hgh = f_means + standard_deviations, f_means - standard_deviations
    f_means.plot(marker='o', label='V^F(s_0)', color='steelblue')
    plt.fill_between(f_means.index, np.where(f_low < 0, 0, f_low), f_hgh, alpha=0.2, color='steelblue')

    plt.xlabel('Epsilon Coefficient')

    plt.ylabel('Forecast')
    plt.title(env_name)
    file_path = path_to.experiment_visuals_folder(env_name) + "epsilon_2"
    plt.savefig(FS.clear_a_path_for(file_path, overwrite=True))
