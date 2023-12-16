n_neurons = 4
import numpy as np
from numpy.random import default_rng
rng = default_rng(seed=111)

def simulate_timeseries(model, noise, T):
    """Simulate time series from different models and ground truth adjacencies for time series causal graph.
    
    :param model: Model for simulation. It take one of the these values: 'lingauss1' for Linear Gaussian Model 1, 'lingauss2' for Linear Gaussian Model 2, 'nonlinnongauss1' for Non-linear Non-Gaussian Model 1, 'nonlinnongauss2' for Non-linear Non-Gaussian Model 2, 'ctrnn': CTRNN
    :type model: string
    :param noise: Noise standard deviation in the simulation
    :type noise: float
    :param T: Number of time points to have in the simulated time series
    :type T: int
    :returns: a tuple of three numpy.array as follows. first numpy.array : Simulated time series of shape (4,T) with T time-recordings for 4 variables. second numpy.array : Grouth truth unweighted adjacency matrix: (i,j) entry represents causal influence from i->j. third numpy.array : Grouth truth weighted adjacency matrix: (i,j) entry has the strength of causal influence from i->j.
    :rtype: tuple
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

def simulate_ctrnn(T,n_vars,w,tau,noise=1):
    """Simulate a continuous time recurrent neural network (CTRNN) time series
    
    :param T: Number of time points to have in the simulated time series
    :type T: int
    :param n_vars: Number of variables to have in the simulated time series
    :type n_vars: int
    :param w: Weights of the CTRNN
    :type w: numpy.array
    :param tau: Time constant of the CTRNN
    :type tau: float
    :param noise: Noise standard deviation in the simulation
    :type noise: float 
    :returns: Simulated time series of shape (n_vars,T) with T time-recordings for n_vars many variables
    :rtype: numpy.array
    """  
    import numpy as np
    e=np.exp(1)
    u=np.zeros((n_vars,T))

    for n in range(T-1):
        for i in range(n_vars):
            In=np.random.normal(1,noise)
            u[i,(n+1)] = u[i,n] - ((e*u[i,n])/tau) + e*np.sum(w[:,i]*np.tanh(u[:,n]))/tau + e*In/tau
    return u