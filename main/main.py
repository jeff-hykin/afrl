from training.train_agent import Agent
from training.train_coach import CoachClass as Coach
from testing.test_predictive import run_test
from testing.analysis import generate_all_visuals

from info import config, path_to

# NOTE: almost all values are pulled from the info.yaml
#       All values in the info.yaml can be overridden with command line arguments
#       for examples and explainations see:
#       see: https://github.com/jeff-hykin/quik_config_python#command-line-arguments

def full_run(env_name, agent_path, coach_path, csv_path, visuals_path):
    agent = Agent.smart_load(
        env_name=env_name,
        path=agent_path,
    )
    coach = Coach.smart_load(
        path=coach_path,
        agent=agent,
    )
    results = run_test(
        env_name,
        coach=coach,
        csv_path=csv_path,
    )
    generate_all_visuals(
        env_name=env_name,
        csv_path=csv_path,
        output_folder=visuals_path,
    )
    return results


# 
# run for all envs 
# 
if __name__ == "__main__":
    for env_name in config.env_names:
        full_run(
            env_name=env_name,
            agent_path=path_to.agent_model_for(env_name),
            coach_path=path_to.coach_model_for(env_name),
            csv_path=path_to.experiment_csv_for(env_name),
            visuals_path=path_to.experiment_visuals_folder(env_name),
        )