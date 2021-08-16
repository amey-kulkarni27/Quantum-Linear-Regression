import networkx as nx
from collections import defaultdict
import pandas as pd
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
import math
# feature selection
def select_features(X_train, y_train, features):
    fs = SelectKBest(score_func=f_regression, k=features)
    fs.fit(X_train, y_train)
    return fs
    
np.random.seed(42)

N = 100000
d = 32
factor = 4
limit = 80

data = np.random.rand(N, d*factor)
Y = np.random.rand(N)

fs = select_features(data,Y,d)
importance = fs.scores_
temp_imp = importance[:]
temp_imp = sorted(temp_imp, reverse=True)
temp_imp = temp_imp[:d]
comp = temp_imp[-1]
pick = [False for i in range(d * factor)]
for i in range(d * factor):
    if(importance[i] >= comp):
        pick[i] = True

D = factor*d
high = sum(importance)
importance /= high
precision_list = list(map(math.ceil,importance*limit))

data_fs = np.zeros((N, d))
j = 0
p_list = [2]
for i in range(d * factor):
    if(pick[i]):
        data_fs[:, j] = data[:, i]
        if precision_list[i] == 1:
            p_list.append(2)
        else:
            p_list.append(precision_list[i])
        j += 1

assert(d + 1 == len(p_list))
for i in range(d + 1):
    assert(p_list[i] != 0)

dim = sum(p_list)
print(p_list)
print(dim)

# Prefix sum of precision
pref_prec = [0]
for i in range(len(p_list)):
    pref_prec.append(pref_prec[-1] + p_list[i])


X = np.ones((N, d + 1))
X[:, :-1] = data_fs
X_sk = np.ones((N, factor*d + 1))
X_sk[:, :-1] = data
XtX = np.matmul(X.T, X)
XtY = np.matmul(X.T, Y)

# Creating the graph
G = nx.Graph()
G.add_edges_from([(i, j) for i in range(dim) for j in range(i + 1, dim)])

# The matrix where we add the objective and the constraint
Q = defaultdict(int)

# Now consider the objective function. It is one big function divided into different blocks
# he weights are [0.25, 0.5, 1 ...] depending on the precision bits offered
start = 3

# First term, same weights
for i in range(d + 1):
    xii = XtX[i, i]
    precision = pref_prec[i + 1] - pref_prec[i]
    for k in range(precision):
        d1 = pref_prec[i] + k
        Q[(d1, d1)] += xii * pow(2, 2 * (k - start))
        for l in range(k + 1, precision):
            d2 = pref_prec[i] + l
            Q[(d1, d2)] += 2 * xii * pow(2, (k - start) + (l - start))

# First term, different weights
for i in range(d + 1):
    for j in range(i + 1, d + 1):
        xij = XtX[i, j]
        precision1 = pref_prec[i + 1] - pref_prec[i]
        precision2 = pref_prec[j + 1] - pref_prec[j]
        for k in range(precision1):
            for l in range(precision2):
                d1 = pref_prec[i] + k
                d2 = pref_prec[j] + l
                Q[(d1, d2)] += 2 * xij * pow(2, (k - start) + (l - start))


# Second Term
for i in range(d + 1):
    xyi = XtY[i]
    precision = pref_prec[i + 1] - pref_prec[i]
    for k in range(precision):
        d1 = pref_prec[i] + k
        Q[(d1, d1)] -= 2 * xyi * pow(2, k - start)


sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample_qubo(Q, num_reads=1000, chain_strength=9)

# Print the entire sampleset, that is, the entire table

# print(sampleset.first.sample)
distributions = []

for sample, energy in sampleset.data(['sample', 'energy']):
    distributions.append(sample)

sol_no = 1
for di in distributions:
    wts = np.array([0.0 for i in range(d + 1)])
    for j in range(d + 1):
        precision = pref_prec[j + 1] - pref_prec[j]
        for k in range(precision):
            wts[j] += di[pref_prec[j] + k] * pow(2, k - start)
    if sol_no == 1:
        # print(str(sol_no) + "-")
        Y_pred = np.matmul(X, wts)
        err = mse(Y, Y_pred)
        print("Quantum Error: ", err)
        print("Quantum Weights:", wts)
        sol_no += 1

clf = LinearRegression()
clf.fit(X_sk, Y)
# print("Classical Sklearn Weights:", clf.coef_)
print("Classical Sklearn Error: ",mse(clf.predict(X_sk),Y))
