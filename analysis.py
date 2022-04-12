from statsmodels.stats.weightstats import ttest_ind
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybullet_envs
import scipy.stats as st
import seaborn as sns
import silver_spectacle as ss
from rigorous_recorder import RecordKeeper, ExperimentCollection

from info import path_to, config
import file_system_py as FS

settings = config.gym_env_settings

sns.set_theme(style="whitegrid")

def confidence_interval(xs):
    return st.t.interval(0.95, len(xs)-1, loc=np.mean(xs), scale=st.sem(xs))

def plot_epsilon_1(env_name, csv_path, output_folder):
    max_reward_single_timestep = settings[env_name].max_reward_single_timestep
    min_reward_single_timestep = settings[env_name].min_reward_single_timestep
    gamma     = settings[env_name].agent_discount_factor
    
    if not FS.exists(csv_path):
        print(f"no data found for: {env_name}: {csv_path}")
        return
    
    data_frame = pd.read_csv(csv_path)
    
    max_reward_single_timestep = data_frame.groupby('epsilon').discounted_rewards.mean().values[0]
    score_range = max_reward_single_timestep - min_reward_single_timestep
    data_frame['normalized_rewards'] = (data_frame.discounted_rewards - min_reward_single_timestep) / score_range
    data_frame['epsilon_adjusted'] = data_frame.epsilon / score_range
    epsilon_means       = data_frame.groupby('epsilon_adjusted').normalized_rewards.mean()
    standard_deviations = data_frame.groupby('epsilon_adjusted').normalized_rewards.std().values
    epsilon_low         = data_frame.epsilon_adjusted.min()
    epsilon_high        = data_frame.epsilon_adjusted.max()
    epsilon             = np.linspace(0, epsilon_high, 2)
    subopt              = epsilon / (1 - gamma)
    base_mean           = 1
    
    plt.figure(figsize=(5.5, 5.5))
    plt.plot(epsilon, base_mean-subopt, color='orange', label='Perform. Bounds')
    epsilon_means.plot(label='V^{PAC}(s_0)', marker='o', ax=plt.subplot(1,1,1), color='steelblue')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Epsilon Coefficient')
    plt.fill_between(epsilon_means.index, epsilon_means-standard_deviations, epsilon_means+standard_deviations, alpha=0.2, color='steelblue')
    plt.hlines(base_mean, epsilon_low, epsilon_high, color='green', label='V^pi(s_0)')
    plt.plot([epsilon_low, epsilon_high], [0,0], color='red', label='V^{rand}(s_0)')
    plt.legend(loc='lower left', fontsize="x-small")
    plt.title(env_name)
    plt.tight_layout()
    
    plt.savefig(
        FS.clear_a_path_for(
            f"{output_folder}/epsilon_1",
            overwrite=True
        )
    )

def plot_epsilon_2(env_name, csv_path, output_folder):
    max_reward_single_timestep = settings[env_name].max_reward_single_timestep
    min_reward_single_timestep = settings[env_name].min_reward_single_timestep
    gamma     = settings[env_name].agent_discount_factor
    
    data_frame = pd.read_csv(csv_path).rename(columns={'Unnamed: 0': 'episode'})
    score_range = max_reward_single_timestep - min_reward_single_timestep
    data_frame['normalized_rewards'] = (data_frame.discounted_rewards - min_reward_single_timestep) / score_range
    data_frame['epsilon_adjusted']   = data_frame.epsilon / score_range
    forecast_means      = data_frame.groupby('epsilon_adjusted').forecast.mean() + 1
    standard_deviations = data_frame.groupby(['epsilon_adjusted', 'episode']).forecast.mean().unstack().std(1)
    forecast_low  = forecast_means + standard_deviations
    forecast_high = forecast_means - standard_deviations
    
    plt.figure(figsize=(5.5, 5.5))
    forecast_means.plot(marker='o', label='V^F(s_0)', color='steelblue')
    plt.fill_between(forecast_means.index, np.where(forecast_low < 0, 0, forecast_low), forecast_high, alpha=0.2, color='steelblue')
    plt.xlabel('Epsilon Coefficient')
    plt.ylabel('Forecast')
    plt.title(env_name)
    plt.savefig(
        FS.clear_a_path_for(
            f"{output_folder}/epsilon_2",
            overwrite=True
        )
    )

def generate_all_visuals(env_name, csv_path, output_folder):
    plot_epsilon_1(
        env_name=env_name,
        csv_path=csv_path,
        output_folder=output_folder,
    )
    plot_epsilon_2(
        env_name=env_name,
        csv_path=csv_path,
        output_folder=output_folder,
    )
    print(f"All visuals saved in {output_folder}")
    