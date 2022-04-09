# Tooling Setup

Everything is detailed in the `documentation/setup.md`!

# Running Code

- Run `python main/main.py -- experiment_name:experiment1` to 
    - train the actor, critic
    - followed by training the coach
    - followed by testing them
    - followed by running analysis, which saves images to the data/visuals folder


To customize it, look at the `info.yaml` file. Find where it has `(default):` and all those are parameters that can be overridden.

For example, lets change the number of training epochs for the coach (aka dynamics). <br>