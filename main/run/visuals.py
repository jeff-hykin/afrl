from info import config, path_to
from analysis import generate_all_visuals
from main.agent import Agent
from main.coach import CoachClass as Coach
from main.test_prediction import run_test

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
        env_name=env_name,
        path=coach_path,
        agent=agent,
    )
    generate_all_visuals(
        env_name=env_name,
        csv_path=csv_path,
        output_folder=visuals_path,
    )


# 
# run for all envs 
# 
if __name__ == "__main__":
    env_name = config.env_name
    print(f'''\n\n-------------------------------------------------------''')
    print(f'''''')
    print(f''' Environment: {env_name}''')
    print(f'''''')
    print(f'''-------------------------------------------------------\n\n''')
    full_run(
        env_name=env_name,
        agent_path=config.load.agent_path or path_to.agent_model_for(env_name),
        coach_path=config.load.coach_path or path_to.coach_model_for(env_name),
        csv_path=path_to.experiment_csv_for(env_name),
        visuals_path=path_to.experiment_visuals_folder(env_name),
    )