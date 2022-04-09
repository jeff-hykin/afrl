# Tooling Setup

Everything is detailed in the `documentation/setup.md`!

# Running Code

- Run `python main/main.py -- experiment_name:experiment1` to 
    - train the actor, critic
    - followed by training the coach
    - followed by testing them
    - followed by running analysis, which saves images to the data/visuals folder


# Customizing  Code

As an example, lets change the number of training epochs for the dynamics model. 

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
            
            train_dynamics:
                number_of_epochs: 100
```

Lets change the iterations from `100` to `500` in the command line.

```sh
python main/main.py -- \
    experiment_name:experiment2 \
    train_dynamics:number_of_epochs:500
```

Thats all there is too it. Works with lists, strings and more advanced types. <br>

See more info at [Quik Config Python](https://github.com/jeff-hykin/quik_config_python#command-line-arguments)
