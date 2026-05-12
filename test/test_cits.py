#%%
from cits import methods
import time
import numpy as np
from numpy.random import default_rng
rng = default_rng(seed=111)
#%%A - model 1
T=1000
n_neurons = 4
noise = 1
alpha = 0.05

A = np.zeros((n_neurons,n_neurons))
A[0,2] = 2
A[1,2] = 1
A[2,3] = 2

def generate_timeseries_A():
    smspikes=np.zeros((n_neurons,T))
    lag=1
    for iter1 in range(n_neurons):
        smspikes[iter1,0]=rng.normal(scale=noise)
    for t in range(1,T):
        smspikes[0,t]=rng.normal(scale=noise)+1
        smspikes[1,t]=rng.normal(scale=noise)-1
        smspikes[2,t]=A[0,2]*np.sum(smspikes[0,np.max((t-lag,0)):t])+A[1,2]*np.sum(smspikes[1,np.max((t-lag,0)):t])+rng.normal(scale=noise)
        smspikes[3,t]=A[2,3]*np.sum(smspikes[2,np.max((t-lag,0)):t])+rng.normal(scale=noise)
        if n_neurons > 4:
            for t1 in range(4,n_neurons,4):
                smspikes[t1,t]=3*np.sum(smspikes[t1-1,np.max((t-lag,0)):t])+rng.normal(scale=noise)+1
                smspikes[t1+1,t]=rng.normal(scale=noise)-1
                smspikes[t1+2,t]=2*np.sum(smspikes[t1,np.max((t-lag,0)):t])+np.sum(smspikes[t1+1,np.max((t-lag,0)):t])+rng.normal(scale=noise)
                smspikes[t1+3,t]=2*np.sum(smspikes[t1+2,np.max((t-lag,0)):t])+rng.normal(scale=noise)
    X = smspikes
    return X
#%%
print("True weighted adjacency matrix: \n")
print(A)
lag=1
A_cf_iter_fin=[]
A_iter_fin=[]
A_cf2_iter_fin=[]   
startime = time.time()
X = generate_timeseries_A()
adj_matrix, causaleff = methods.cits_full_weighted(X,lag,alpha)
out = str(time.time()-startime)
print("time taken "+ out + "\n")

print("Estimated weighted adjacency matrix: \n")
print(causaleff)

#%%
print("True unweighted adjacency matrix: \n")
print((A!=0).astype(int))
lag=1
A_cf_iter_fin=[]
A_iter_fin=[]
A_cf2_iter_fin=[]   
startime = time.time()
X = generate_timeseries_A()
adj_matrix = methods.cits_full(X,lag,alpha)
out = str(time.time()-startime)
print("time taken "+ out + "\n")

print("Estimated unweighted adjacency matrix: \n")
print(adj_matrix)

# %%
# Regression test for v1.4 fixes:
#   1. partial_corr now fits an intercept -> shift-invariant
#   2. cits_unrolled conditioning set spans the full unrolled graph and
#      excludes the two variables being tested
#
# Test by shifting the time series by large constants. With the intercept
# fix, the recovered graph and weighted effects should be identical to
# the unshifted run; without the fix, shifting biases the residuals and
# can change the recovered graph.
print("\n=== v1.4 regression test: shift-invariance of partial_corr ===")
X_unshifted = generate_timeseries_A()
adj_un, eff_un = methods.cits_full_weighted(X_unshifted, lag, alpha)

shifts = np.array([[100.0], [-50.0], [1000.0], [0.0]])  # per-neuron shifts
X_shifted = X_unshifted + shifts
adj_sh, eff_sh = methods.cits_full_weighted(X_shifted, lag, alpha)

print("Adjacency identical (shifted vs unshifted): ",
      np.array_equal(adj_un, adj_sh))
print("Weighted effects close (atol=1e-6):       ",
      np.allclose(eff_un, eff_sh, atol=1e-6))
assert np.array_equal(adj_un, adj_sh), \
    "Shift-invariance broken: adjacency matrices differ"
assert np.allclose(eff_un, eff_sh, atol=1e-6), \
    "Shift-invariance broken: weighted effects differ"
print("v1.4 regression test PASSED")

