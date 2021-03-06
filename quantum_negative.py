import networkx as nx
from collections import defaultdict
import pandas as pd
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression

np.random.seed(42)

N = 1000
d = 4
precision = 6
dim = (d + 1) * (2 * precision)

def multiplier(x):
    if x <= 2:
        return 1
    else:
        return -1

data = np.random.rand(N, d)
Y = np.random.rand(N)
# Y = 0.5 * data[:, 0] + 1.25 * data[:, 1]
X = np.ones((N, d + 1))
X[:, :-1] = data
XtX = np.matmul(X.T, X)
XtY = np.matmul(X.T, Y)

# Creating the graph
G = nx.Graph()
G.add_edges_from([(i, j) for i in range(dim) for j in range(i + 1, dim)])

# The matrix where we add the objective and the constraint
Q = defaultdict(int)

# Now consider the objective function. It is one big function divided into different blocks
# For now, the weights are [1, 0.5, 0.25, -1, -0.5, -0.25] depending on the precision bits required

# First term, same weights
for i in range(d + 1):
    xii = XtX[i, i]
    for k in range(2 * precision):
        d1 = i * 2 * precision + k
        Q[(d1, d1)] += xii / pow(2, 2 * (k % precision))
        for l in range(k + 1, 2 * precision):
            d2 = i * 2 * precision + l
            Q[(d1, d2)] += 2 * xii / pow(2, k % precision + l % precision) * multiplier(k) * multiplier(l)

# First term, different weights
for i in range(d + 1):
    for j in range(i + 1, d + 1):
        xij = XtX[i, j]
        for k in range(2 * precision):
            for l in range(2 * precision):
                d1 = i * 2 * precision + k
                d2 = j * 2 * precision + l
                Q[(d1, d2)] += 2 * xij / pow(2, k % precision + l % precision) * multiplier(k) * multiplier(l)


# Second Term
for i in range(d + 1):
    xyi = XtY[i]
    for k in range(2 * precision):
        d1 = i * 2 * precision + k
        Q[(d1, d1)] -= 2 * xyi / pow(2, k % precision) * multiplier(k)


sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample_qubo(Q, num_reads=1500, chain_strength=10)

# Print the entire sampleset, that is, the entire table
# print(sampleset)

distributions = []

for sample, energy in sampleset.data(['sample', 'energy']):
    distributions.append(sample)

sol_no = 1
for di in distributions:
    wts = np.array([0.0 for i in range(d + 1)])
    for x in range(dim):
        i = x // (2 * precision)
        k = x % (2 * precision)
        wts[i] += di[x] / pow(2, k % precision) * multiplier(k)
    if sol_no == 1:
        print(str(sol_no) + "-")
        Y_pred = np.matmul(X, wts)
        err = mse(Y, Y_pred)
        print("Error: ", err)
        print("Weights:", wts)
        sol_no += 1

clf = LinearRegression()
clf.fit(X, Y)
print(clf.coef_)
print("MSE Sklearn: ",mse(clf.predict(X),Y))
