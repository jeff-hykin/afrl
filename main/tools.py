def flatten(ys):
    return [x for xs in ys for x in xs]

def get_discounted_rewards(rewards, gamma):
    return sum([r * gamma ** t for t, r in enumerate(rewards)])

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
        self.index = 0
        self.steps = []
    
    @property
    def prev(self):
        if self.index > 0:
            return self.steps[-1]
        else:
            return None
    
    def next(self, prev_state, action, reward, state):
        self.index += 1
        self.steps.append(Timestep(self.index, prev_state, action, reward, state))
    
    def rewards(self):
        return [ each.reward for each in self.steps ]
    
    def curr_and_next_states(self):
        return [ each.prev_state for each in self.steps ], [ each.state for each in self.steps ]
    
    def items(self):
        """
        for index, state, action, reward, next_state in time_series.items():
            pass
        """
        return ((each.index, each.prev_state, each.action, each.reward, each.state) for each in self.steps)
    
    def __getitem__(self, key):
        time_slice =  TimestepSeries()
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