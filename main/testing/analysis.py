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
    plt.figure(figsize=(5.5, 5.5))
    
    min_score = settings[env_name].min_score
    gamma     = settings[env_name].agent_discount_factor
    
    if not FS.exists(csv_path):
        print(f"no data found for: {env_name}: {csv_path}")
        return
    
    data_frame = pd.read_csv(csv_path).rename(columns={'discounted_rewards': 'V'})
    
    epsilon_low   = data_frame.eps.min() 
    epsilon_high  = data_frame.eps.max()
    epsilon_means = data_frame.groupby('eps').opt.mean()
    epsilon_stdev = data_frame.groupby('eps').opt.std().values
    max_score     = data_frame.groupby('epsilon').V.mean().values[0]
    
    score_range = max_score - min_score
    
    data_frame['opt'] = (data_frame.V - min_score) / score_range
    data_frame['eps'] = data_frame.epsilon / score_range
    
    epsilon = np.linspace(0, epsilon_high, 2)
    subopt  = epsilon / (1 - gamma)
    epsilon_means.plot(
        label='V^{PAC}(s_0)',
        marker='o',
        ax=plt.subplot(1,1,1),
        color='steelblue'
    )
    base_mean = 1
    plt.plot(epsilon, base_mean-subopt, color='orange', label='Perform. Bounds')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Epsilon Coefficient')
    plt.fill_between(epsilon_means.index, epsilon_means-epsilon_stdev, epsilon_means+stdev, alpha=0.2, color='steelblue')
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
    max_score = settings[env_name]['max_score']
    min_score = settings[env_name]['min_score']
    
    data_frame = pd.read_csv(csv_path).rename(columns={'discounted_rewards': 'V', 'Unnamed: 0': 'episode'})
    forecast_means               = data_frame.groupby('eps').forecast.mean() + 1
    forecast_standard_deviations = data_frame.groupby(['eps', 'episode']).forecast.mean().unstack().std(1)
    
    score_range = max_score - min_score
    data_frame['opt'] = (data_frame.V - min_score) / score_range
    data_frame['eps'] = data_frame.epsilon / score_range
    
    forecast_lows   = forecast_means + forecast_standard_deviations # NOTE: this seems backwards to me, not that the graph will look any different --Jeff
    forecasts_highs = forecast_means - forecast_standard_deviations
    forecast_means.plot(marker='o', label='V^F(s_0)', color='steelblue')
    
    plt.figure(figsize=(5.5, 5.5))
    plt.fill_between(forecast_means.index, np.where(forecast_lows < 0, 0, forecast_lows), forecasts_highs, alpha=0.2, color='steelblue')
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
    