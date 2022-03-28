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