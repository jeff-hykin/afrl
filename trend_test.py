import numpy as np
data = [1,2,4,3,6,5,8,7,10,9,12,11]
N = len(data)
timestamps = np.arange(1, N+1)
increasing_pairs = [data[i] > data[j] for j in range(N) for i in range(j+1, N)]
decreasing_pairs = [data[i] < data[j] for j in range(N) for i in range(j+1, N)]
C = np.sum(increasing_pairs)
D = np.sum(decreasing_pairs)
T = (C-D)/(C+D)
z = (3 * T * np.sqrt(N*(N-1))) / np.sqrt(2*(2*N+5))
df = N - 2
print(z)
from scipy.stats import norm
print((1-norm.cdf(z))*2)