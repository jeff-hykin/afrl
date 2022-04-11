# Tooling Setup

Everything is detailed in the `documentation/setup.md`!

# Running Code

- Run `python main/main.py -- experiment_name:experiment1` to 
    - train the actor, critic
    - followed by training the coach
    - followed by testing them
    - followed by running analysis, which saves images to the data/visuals folder


# Customizing  Code

As an example, lets change the number of training epochs for the coach model. 

- Take a look at the `info.yaml` file.
- Find where it has `(default):`

Example look at the `info.yaml` file:

```yaml
(project):
    
    ###stuff###
    
    (profiles):
        
        ###stuff###
        
        (default):
            
            ###stuff###
            
            train_coach:
                number_of_epochs: 100
```

Lets change the iterations from `100` to `500` in the command line.

```sh
python main/main.py -- \
    experiment_name:experiment2 \
    train_coach:number_of_epochs:500
```

Thats all there is too it. Works with lists, strings and more advanced types. <br>

Full example:

```sh
python main/main.py --                                            \
    experiment_name:experiment3                                   \
    env_name:AntBulletEnv-v0                                      \
    agent_path:./data/models/agents/AntBulletEnv-v0/default_sac_1 \
    train_coach:loss_style:timestep                               \
    train_coach:loss_function:consistent_coach_loss               
```

#### See more info at [Quik Config Python](https://github.com/jeff-hykin/quik_config_python#command-line-arguments)
