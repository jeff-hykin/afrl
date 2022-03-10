import numpy as np
from scipy.stats import norm


def trend_test(data):
    N = len(data)
    increasing_pairs = [data[i] > data[j] for j in range(N) for i in range(j + 1, N)]
    decreasing_pairs = [data[i] < data[j] for j in range(N) for i in range(j + 1, N)]
    C = np.sum(increasing_pairs)
    D = np.sum(decreasing_pairs)
    T = (C - D) / (C + D)
    # print('C, D, T', C, D, T, N)
    z = (3 * T * np.sqrt(N * (N - 1))) / np.sqrt(2 * (2 * N + 5))
    # print('z', z)
    # df = N - 2
    pval = (1 - norm.cdf(np.abs(z))) * 2
    return pval


if __name__ == "__main__":
    # data = [1,2,4,3,6,5,8,7,10,9,12,11]
    data = np.load("data/results/predpolicy/LunarLanderContinuous-v2.npy")
    trend_test(data)
