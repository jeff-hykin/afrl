from info import config, path_to
from analysis import generate_all_visuals
from main.agent import Agent
from main.coach import Coach as Coach


settings = config.agent_compare_test
env_name = config.env_name
env = config.get_env(env_name)

print(f'''\n\n-------------------------------------------------------''')
print(f'''''')
print(f''' Environment: {env_name}''')
print(f'''''')
print(f'''-------------------------------------------------------\n\n''')

agents = {}
for agent_index in range(settings.number_of_agents):
    config.train_agent.model_name = f'{settings.base_name}_{agent_index+1}'
    agents[config.train_agent.model_name] = Agent.smart_load(
        env_name=env_name,
        path=path_to.agent_model_for(env_name),
    )
    agents[config.train_agent.model_name].gather_experience(env=env, number_of_episodes=settings.number_of_episodes)
    

print(f'''\n\n-------------------------------------------------------''')
for each_agent_name, each_agent in agents.items():
    print(f'''Agent {each_agent_name}, average_reward:{each_agent.average_reward}''')
print(f'''-------------------------------------------------------\n\n''')