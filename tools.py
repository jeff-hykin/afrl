def average(iterable):
    from statistics import mean
    from trivial_torch_tools.generics import to_pure
    return mean(tuple(to_pure(each) for each in iterable))

def median(iterable):
    from statistics import median
    from trivial_torch_tools.generics import to_pure
    return median(tuple(to_pure(each) for each in iterable))

def to_numpy(value):
    import torch
    import numpy
    from trivial_torch_tools.generics import to_pure
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    else:
        return numpy.array(to_pure(value))

def flatten(ys):
    return [x for xs in ys for x in xs]

def get_discounted_rewards(rewards, gamma):
    return sum([r * gamma ** t for t, r in enumerate(rewards)])

def normalize_rewards(rewards, max_reward_single_timestep, min_reward_single_timestep):
    """
    all elements of the output should be between 0 and 1
    """
    reward_range = max_reward_single_timestep - min_reward_single_timestep
    return tuple((each - min_reward_single_timestep)/reward_range for each in rewards)

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]

def minibatch(batch_size, *data):
    import numpy as np
    indices = np.arange(len(data[0]))
    np.random.shuffle(indices)
    for batch_ind in divide_chunks(indices, batch_size):
        yield [datum[batch_ind] for datum in data]

def bundle(iterable, bundle_size):
    next_bundle = []
    for each in iterable:
        next_bundle.append(each)
        if len(next_bundle) >= bundle_size:
            yield tuple(next_bundle)
            next_bundle = []
    # return any half-made bundles
    if len(next_bundle) > 0:
        yield tuple(next_bundle)

def rolling_average(a_list, window):
    results = []
    if len(a_list) < window * 2:
        return a_list
    near_the_end = len(a_list) - 1 - window 
    for index, each in enumerate(a_list):
        # at the start
        if index < window:
            average_items = a_list[0:index]+a_list[index:index+window]
        # at the end
        elif index > near_the_end:
            average_items = a_list[index-window:index]+a_list[index:len(a_list)]
        else:
            # this could be done a lot more efficiently with a rolling sum, oh well! ¯\_(ツ)_/¯ 
            average_items = a_list[index-window:index+window+1]
        results.append(sum(average_items)/len(average_items))
    return results

def ft(arg):
    from torch import FloatTensor
    from info import config
    return FloatTensor(arg).to(config.device)

def train_test_split(*args, split_proportion):
    import numpy as np
    def split(data, indices, train_pct=0.66):
        div = int(len(data) * train_pct)
        train, test = indices[:div], indices[div:]
        return data[train], data[test]
    
    indices = np.arange(len(args[0]))
    np.random.shuffle(indices)
    output = []
    for each in args:
        output.append(split(each, indices, split_proportion))
    
    return output

colors = dict(
    yellow=       '#fec355',
    light_yellow= '#ddd790',
    lime=         '#c3e88d',
    green=        '#4ec9b0',
    light_blue=   '#89ddff',
    blue=         '#82aaff',
    deep_blue=    '#00aeff',
    purple=       '#c792ea',
    pink=         '#e57eb3',
    red=          '#f07178',
)
def wrap_around_get(number, a_list):
    list_length = len(a_list)
    return a_list[((number % list_length) + list_length) % list_length]

def get_color(index):
    return wrap_around_get(index, list(colors.values()))

def multi_line_plot(a_dict, path, x_axis_label, y_axis_label):
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    for index, (line_name, line_points) in enumerate(a_dict.items()):
        x_values = tuple(x for x,y in line_points)
        y_values = tuple(y for x,y in line_points)
        color = get_color(index=index)
        plt.plot(x_values, y_values, marker='.', color=color, label=line_name)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()
    plt.savefig(
        FS.clear_a_path_for(
            path,
            overwrite=True
        )
    )
    return plt

from torch import nn
def feed_forward(layer_sizes, activation=nn.Tanh, output_activation=nn.Identity):
    from trivial_torch_tools import Sequential
    layers = []
    for index in range(len(layer_sizes) - 1):
        activation_class = activation if index < len(layer_sizes) - 2 else output_activation
        layers += [
            nn.Linear(layer_sizes[index], layer_sizes[index + 1]),
            activation_class(),
        ]
    return Sequential(*layers)

from dataclasses import dataclass, field
@dataclass
class Timestep:
    index      : int
    prev_state : None
    action     : None
    reward     : float
    state      : None
        
class TimestepSeries:
    def __init__(self, ):
        self.index = -1
        self.steps = []
    
    @property
    def prev(self):
        if self.index > 0:
            return self.steps[-1]
        else:
            return None
    
    def add(self, prev_state, action=None, reward=None, state=None):
        # if timestep, pull all the data out of the timestep
        if isinstance(prev_state, Timestep):
            action = prev_state.action
            reward = prev_state.reward
            state = prev_state.state
            prev_state = prev_state.prev_state
            
        self.index += 1
        self.steps.append(Timestep(self.index, prev_state, action, reward, state))
    
    @property
    def states(self):
        return [ each.prev_state for each in self.steps ]
    
    @property
    def actions(self):
        return [ each.action for each in self.steps ]
    
    @property
    def rewards(self):
        return [ each.reward for each in self.steps ]
    
    @property
    def next_states(self):
        return [ each.state for each in self.steps ]
    
    def items(self):
        """
        for index, state, action, reward, next_state in time_series.items():
            pass
        """
        return ((each.index, each.prev_state, each.action, each.reward, each.state) for each in self.steps)
    
    def __len__(self):
        return len(self.steps)
        
    def __getitem__(self, key):
        time_slice = TimestepSeries()
        time_slice.index = self.index
        time_slice.steps = self.steps[key]
        return time_slice
    
    def __repr__(self):
        string = "TimestepSeries(\n"
        for index, state, action, reward, next_state in self.items():
            string += f"    {index}, {state}, {action}, {reward}, {next_state}\n"
        string += ")"
        return string


from dataclasses import dataclass, field
@dataclass
class Episode:
    states     : list = field(default_factory=list)
    actions    : list = field(default_factory=list)
    rewards    : list = field(default_factory=list)
    reward_total: float = 0
    
    @property
    def curr_states(self):
        return [s for s in self.states[1:  ]]
    
    @property
    def next_states(self):
        return [s for s in self.states[: -1]]