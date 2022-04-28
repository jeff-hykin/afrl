def average(iterable):
    from statistics import mean
    from trivial_torch_tools.generics import to_pure
    return mean(tuple(to_pure(each) for each in iterable))

def median(iterable):
    from statistics import median
    from trivial_torch_tools.generics import to_pure
    return median(tuple(to_pure(each) for each in iterable))

def stats(number_iterator):
    import math
    from super_map import LazyDict
    from statistics import stdev, median, quantiles
    from trivial_torch_tools.generics import to_pure
    
    minimum = math.inf
    maximum = -math.inf
    total = 0
    values = [] # for iterables that get consumed
    for each in number_iterator:
        values.append(to_pure(each))
        total += each
        if each > maximum:
            maximum = each
        if each < minimum:
            minimum = each
    
    count = len(values)
    range = maximum-minimum
    average     = total / count     if count != 0 else None
    median      = median(values)    if count != 0 else None
    stdev       = stdev(values)     if count  > 1 else None
    normalized  = tuple((each-minimum)/range for each in values) if range != 0 else None
    (q1,_,q3),_ = quantiles(values) if count  > 1 else (None,None,None),None
    
    return LazyDict(
        max=maximum,
        min=minimum,
        range=range,
        count=count,
        sum=total,
        average=average,
        stdev=stdev,
        median=median,
        q1=q1,
        q3=q3,
        normalized=normalized,
    )    

def simple_stats(number_iterator):
    import math
    from super_map import LazyDict
    from statistics import stdev, median, quantiles
    from trivial_torch_tools.generics import to_pure
    
    minimum = math.inf
    maximum = -math.inf
    total = 0
    values = [] # for iterables that get consumed
    for each in number_iterator:
        values.append(to_pure(each))
        total += each
        if each > maximum:
            maximum = each
        if each < minimum:
            minimum = each
    
    count = len(values)
    range = maximum-minimum
    average     = total / count     if count != 0 else None
    median      = median(values)    if count != 0 else None
    stdev       = stdev(values)     if count  > 1 else None
    
    return LazyDict(
        max=maximum,
        min=minimum,
        range=range,
        count=count,
        sum=total,
        average=average,
        stdev=stdev,
        median=median,
    )    

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
    import torch
    from trivial_torch_tools import to_tensor
    rewards   = to_tensor(rewards)
    timesteps = to_tensor(range(len(rewards)))
    gammas    = to_tensor(gamma for each in timesteps)
    return rewards * (gammas ** timestep)

def normalize(values, max, min):
    """
    all elements of the output should be between 0 and 1
    """
    reward_range = max - min
    return tuple((each - min)/reward_range for each in values)

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
        # fallback
        if len(average_items) == 0:
            average_items = [ a_list[index] ]
        results.append(sum(average_items)/len(average_items))
    return results

def ft(arg):
    from torch import FloatTensor
    from info import config, print
    return FloatTensor(arg).to(config.device)

def log_scale(number):
    import math
    if number > 0:
        return math.log(number+1)
    else:
        return -math.log((-number)+1)

def train_test_split(*args, split_proportion):
    import numpy as np
    from trivial_torch_tools import to_tensor
    def split(data, indices, train_pct=0.66):
        div = int(len(data) * train_pct)
        train, test = indices[:div], indices[div:]
        return to_tensor(data[train]), to_tensor(data[test])
    
    indices = np.arange(len(args[0]))
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

def key_prepend(key, a_dict):
    """
    key_prepend("reward", stats(rewards))
    # { "max": , "min": , ... }
    # =>
    # { "reward_max": , "reward_min": , ... }
    """
    new_dict = {}
    for each_key, each_value in a_dict.items():
        new_key = f"{key}_{each_key}"
        new_dict[new_key] = each_value
    return new_dict

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

def log_graph(data):
    import silver_spectacle as ss
    colors = [
        'rgb(75, 192, 192, 0.5)',
        'rgb(0, 292, 192, 0.5)',
        'rgb(0, 92, 192, 0.5)',
        'rgb(190, 92, 192, 0.5)',
        'rgb(75, 192, 192, 0.5)',
        'rgb(0, 292, 192, 0.5)',
        'rgb(0, 92, 192, 0.5)',
        'rgb(190, 92, 192, 0.5)',
        'rgb(75, 192, 192, 0.5)',
        'rgb(0, 292, 192, 0.5)',
        'rgb(0, 92, 192, 0.5)',
        'rgb(190, 92, 192, 0.5)',
        'rgb(75, 192, 192, 0.5)',
        'rgb(0, 292, 192, 0.5)',
        'rgb(0, 92, 192, 0.5)',
        'rgb(190, 92, 192, 0.5)',
    ]
    ss.DisplayCard("chartjs", {
        "type": 'line',
        "data": {
            "datasets": [
                {
                    "label": each_key,
                    "data": ({"x":x, "y":y} for x,y in each_value),
                    "tension": 0.1,
                    "backgroundColor": colors.pop(),
                }
                    for each_key, each_value in data.items()
            ]
        },
        "options": {
            "pointRadius": 3, # the size of the dots
            "scales": {
                "y": {
                    "type": "logarithmic",
                },
            }
        }
    })

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
        return [s for s in self.states[: -1]]
    
    @property
    def next_states(self):
        return [s for s in self.states[1: ]]


class WeightUpdate(object):
    """
    with WeightUpdate(optimizer=self.optimizer) as step:
        step.loss = self.loss_function()
    """
    def __init__(self, *, optimizer):
        self.optimizer = optimizer
        self.loss = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, _, error, traceback):
        if error is not None:
            # error cleanup HERE
            raise error
            
        if self.optimizer:
            self.optimizer.zero_grad()
            if not (self.loss is None):
                self.loss.backward()
            self.optimizer.step()

def confidence_interval(confidence_percent, samples):
    import statistics
    import scipy.stats as st
    min_value, max_value = st.t.interval(alpha=confidence_percent/100, df=len(samples)-1, loc=statistics.mean(samples), scale=st.sem(samples)) 
    return min_value, max_value

def confidence_interval_size(confidence_percent, samples):
    import statistics
    import scipy.stats as st
    min_value, max_value = st.t.interval(alpha=confidence_percent/100, df=len(samples)-1, loc=statistics.mean(samples), scale=st.sem(samples)) 
    return abs(max_value-min_value)/2
    
def probability_of_belonging_if_bellcurve(item, bellcurve_mean, bellcurve_stdev):
    import scipy.stats as stats
    import math
    how_many_deviations_away = abs(item-bellcurve_mean) / bellcurve_stdev
    return stats.norm.cdf(how_many_deviations_away)

def jenson_shannon_divergence(net_1_logits, net_2_logits):
    from torch.functional import F
    net_1_probs =  F.softmax(net_1_logits, dim=0)
    net_2_probs=  F.softmax(net_2_logits, dim=0)
    
    total_m = 0.5 * (net_1_probs + net_1_probs)
    
    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=0), total_m, reduction="batchmean") 
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=0), total_m, reduction="batchmean") 
    return (0.5 * loss)

def save_all_charts_to(path, overwrite=True):
    import requests
    import file_system_py as FS
    FS.clear_a_path_for(path, overwrite=overwrite)
    FS.write(
        data=requests.get(url='http://localhost:9900/').text,
        to=path,
    )