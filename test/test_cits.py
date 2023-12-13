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
