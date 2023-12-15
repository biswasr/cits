n_neurons = 4
import numpy as np
from numpy.random import default_rng
rng = default_rng(seed=111)

def simulate_timeseries(model, noise, T):
    """Simulate time series from different models and ground truth adjacencies for time series causal graph.
    
    Args:
        model: (string)
            'lingauss1': Linear Gaussian Model 1
            'lingauss2': Linear Gaussian Model 2
            'nonlinnongauss1': Non-linear Non-Gaussian Model 1
            'nonlinnongauss2': Non-linear Non-Gaussian Model 2
            'ctrnn': CTRNN
        noise: (float) Noise standard deviation in the simulation
        T: (int) Number of time points to have in the simulated time series

    Returns:
        X: (numpy.array) Simulated time series of shape (T,4) with T time-recordings for 4 variables
        groundtruthadj: (numpy.array) Grouth truth unweighted adjacency matrix: (i,j) entry represents causal influence from i->j. 
        groundtruthadj_weighted: (numpy.array) Grouth truth weighted adjacency matrix: (i,j) entry has the strength of causal influence from i->j.

    """

    smspikes=np.zeros((n_neurons,T))
    groundtruthadj_weighted = np.zeros((n_neurons,n_neurons))
    lag=1
    if model == 'lingauss1':
        for iter1 in range(n_neurons):
            smspikes[iter1,0]=rng.normal(scale=noise)
        for t in range(1,T):
            smspikes[0,t]=rng.normal(scale=noise)+1
            smspikes[1,t]=rng.normal(scale=noise)-1
            smspikes[2,t]=2*np.sum(smspikes[0,np.max((t-lag,0)):t])+1*np.sum(smspikes[1,np.max((t-lag,0)):t])+rng.normal(scale=noise)
            smspikes[3,t]=2*np.sum(smspikes[2,np.max((t-lag,0)):t])+rng.normal(scale=noise)
        
        groundtruthadj_weighted[0,2] = 2
        groundtruthadj_weighted[1,2] = 1
        groundtruthadj_weighted[2,3] = 2
    
    elif model == 'lingauss2':
        for iter1 in range(n_neurons):
            smspikes[iter1,0]=rng.normal(scale=noise)
        for t in range(1,T):
            smspikes[0,t]=rng.normal(scale=noise)+1
            smspikes[1,t]=-1+2*np.sum(smspikes[0,np.max((t-lag,0)):t])+rng.normal(scale=noise)
            smspikes[2,t]=2*np.sum(smspikes[0,np.max((t-lag,0)):t])+rng.normal(scale=noise)
            smspikes[3,t]=np.sum(smspikes[1,np.max((t-lag,0)):t])+np.sum(smspikes[2,np.max((t-lag,0)):t])+rng.normal(scale=noise)

        groundtruthadj_weighted[0,1] = 2
        groundtruthadj_weighted[0,2] = 2
        groundtruthadj_weighted[1,3] = 1 
        groundtruthadj_weighted[2,3] = 1    
    
    elif model == 'nonlinnongauss1':
        for iter1 in range(n_neurons):
            smspikes[iter1,0]=np.random.uniform()
        for t in range(1,T):
            smspikes[0,t]=np.random.uniform(high = noise)
            smspikes[1,t]=np.random.uniform(high = noise)
            smspikes[2,t]=4*np.sum(np.sin(smspikes[0,np.max((t-lag,0)):t]))-3*np.sum(np.sin(smspikes[1,np.max((t-lag,0)):t]))+np.random.uniform(high = noise)
            smspikes[3,t]=3*np.sum(np.sin(smspikes[2,np.max((t-lag,0)):t]))+np.random.uniform(high = noise)

        groundtruthadj_weighted[0,2] = 1
        groundtruthadj_weighted[1,2] = -1
        groundtruthadj_weighted[2,3] = 1

    elif model == 'nonlinnongauss2':
        for iter1 in range(n_neurons):
            smspikes[iter1,0]=rng.normal(scale=noise)
        for t in range(1,T):
            smspikes[0,t]=rng.normal(scale=noise)
            smspikes[1,t]=4*np.sum(smspikes[0,np.max((t-lag,0)):t])+rng.normal(scale=noise)
            smspikes[2,t]=3*np.sum(np.sin(smspikes[0,np.max((t-lag,0)):t]))+rng.normal(scale=noise)
            smspikes[3,t]=8*np.sum(np.log(np.abs(smspikes[1,np.max((t-lag,0)):t])))+9*np.sum(np.log(np.abs(smspikes[2,np.max((t-lag,0)):t])))+rng.normal(scale=noise)

        groundtruthadj_weighted[0,1] = 1
        groundtruthadj_weighted[0,2] = 1
        groundtruthadj_weighted[1,3] = 1
        groundtruthadj_weighted[2,3] = 1

    elif model == 'ctrnn':
        lag=1
        w=np.zeros((n_neurons,n_neurons))
        w[0,2]=100
        w[1,2]=100
        w[2,3]=100
        tau=10
        smspikes=simulate_ctrnn(T,n_neurons,w,tau,noise)

        groundtruthadj_weighted[0,0] = 1
        groundtruthadj_weighted[1,1] = 1
        groundtruthadj_weighted[2,2] = 1
        groundtruthadj_weighted[3,3] = 1  
        groundtruthadj_weighted[0,2] = 1
        groundtruthadj_weighted[1,2] = 1
        groundtruthadj_weighted[2,3] = 1

    groundtruthadj = (groundtruthadj_weighted!=0).astype(int)
    X = smspikes
    return X, groundtruthadj, groundtruthadj_weighted

def simulate_ctrnn(T,w,tau,noise=1):
    """Simulate a continuous time recurrent neural network (CTRNN) time series
    
    Args:
        T: (int) Number of time points to have in the simulated time series
        w: (numpy.array) Weights of the CTRNN
        tau: (float) Time constant of the CTRNN
        noise: (float) Noise standard deviation in the simulation

    Returns:
        u: (numpy.array) Simulated time series of shape (T,4) with T time-recordings for 4 variables
    
    """  
    import numpy as np
    e=np.exp(1)
    u=np.zeros((m,n_ctrnn))

    for n in range(n_ctrnn-1):
        for i in range(m):
            In=np.random.normal(1,noise)
            u[i,(n+1)] = u[i,n] - ((e*u[i,n])/tau) + e*np.sum(w[:,i]*np.tanh(u[:,n]))/tau + e*In/tau
    return u