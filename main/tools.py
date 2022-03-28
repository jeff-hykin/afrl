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