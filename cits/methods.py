import numpy as np
from itertools import chain, combinations
from scipy import stats, linalg
def data_transform(X,tau):
    """Transforms data X of shape (p,T) to time-windowed samples $\chi$ of shape $(p*2*(\tau+1),N)$, where $N = \lfloor \frac{T-2*(\tau+1)}{2*(\tau+1)} \rfloor$.

    :param X: pxT array with p variables and T time points
    :type X: numpy.array
    :tau: Markovian order, in other words, maximum time delay of interaction, i.e. $X_t$ can depend up to $X_{t-\tau}$ and not earlier.
    :type tau: int
    :returns: Transformed dataset which is an array of shape $(p*2*(\tau+1),N)$
    :rtype: numpy.array 
    """ 
    p = X.shape[0]
    n = X.shape[1]
    N = int((n-2*(tau+1))/(2*(tau+1)))

    chi = np.zeros((p*2*(tau+1),N))

    #transform X to time-windowed samples chi
    for i in range(N):
        chi[:,i] = X[:,2*(tau+1)*i:2*(tau+1)*(i+1)].T.reshape(p*2*(tau+1))
    return chi


def data_transformed(X, tau):
    """Transforms data X of shape (p,n) to time-windowed samples of shape $(N,p*(\tau+1))$, with an inter-sample time gap of $\tau+1$, where $N = \lfloor \frac{n-2*(\tau+1)}{2*(\tau+1)} \rfloor$.

    :param X: pxT array with p variables and T time points
    :type X: numpy.array
    :param tau: Markovian order, in other words, maximum time delay of interaction, i.e. $X_t$ can depend up to $X_{t-\tau}$ and not earlier.
    :type tau: int
    :returns: Transformed dataset which is an array of shape $(N,p*(\tau+1))$
    :rtype: numpy.array
    """
    data = X.T
    n = data.shape[0]
    p = data.shape[1]
    lag = tau
    lag1 = lag+1
    new_n = int(np.floor((n-lag)/(2*lag1))*(2*lag1))
    data=data[:new_n,:]
    data2=np.zeros((int(new_n/(2*lag1)),p*lag1))
    for i in range(p):
        for j in range(lag1):
            data2[:,lag1*i+j]=data[j::(2*lag1),i]
    return data2

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def cond_dep_pcorr(chi,i1,j1,k,p,alpha=0.05):
    """Conditional dependence test based on partial correlation"""
    (v2,t2) = j1
    (v1,t1) = i1
    i = t1*p + v1
    j = t2*p + v2
    r = partial_corr(i,j,set(k),chi)
    if r==1:
        pval = 0    
    else:
        z = 0.5 * np.log((1+r)/(1-r))
        T = np.sqrt(chi.shape[1]-len(k)-3)*np.abs(z)
        pval = 2*(1 - stats.norm.cdf(T))
    if pval<=alpha:
        out = 1
    else:
        out = 0
    return out
def cond_dep_hsic(chi,i1,j1,k,p,alpha=0.05,reps=10):
    """Convenient wrapper for conditional dependence test based on Hilbert-Schmidt criterion
    """
    (v2,t2) = j1
    (v1,t1) = i1
    i = t1*p + v1
    j = t2*p + v2
    pval = hsic_condind(i,j,set(k),chi,reps)
    if pval<=alpha:
        out = 1
    else:
        out = 0
    return out


def partial_corr(A,B,S,data):
    """Computes partial correlation of variables A and B given S based on data. 
    
    :param A: index of one variable
    :type A: int  
    :param B: index of another variable
    :type B: int
    :param S: indices of conditioning variables
    :type S: set
    :param data: Dataset with samples for the variables
    :type data: numpy.array
        shape of data should be (# of variables, # of samples)
    :returns: Partial correlation of A and B given S based on data
    :rtype: float
    """
    p = data.shape[0]
    idx = np.zeros(p, dtype=bool)

    for i in range(p):
        if i in S:
            idx[i]=True
    C=data
    beta_A = linalg.lstsq(C[idx,:].T, C[A,:].T)[0]
    beta_B = linalg.lstsq(C[idx,:].T, C[B,:].T)[0]

    res_A = C[A,:].T - C[idx,:].T.dot(beta_A)
    res_B = C[B,:].T - C[idx,:].T.dot(beta_B)
    
    p_corr = stats.pearsonr(res_A, res_B)[0]  
    
    return p_corr
def hsic_condind(A,B,S,data,reps):
    """Computes p-value of conditional dependence test based on Hilbert-Schmidt criterion 
    """
    import pandas as pd
    import os
    #os.environ["R_HOME"] = r"D:\R-4.2.2"
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    import rpy2.rlike.container as rlc
    from rpy2.robjects import pandas2ri
    data_trans = data.T
    d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    kpcalg = importr('kpcalg', robject_translations = d)
    data_trans_pd=pd.DataFrame(data_trans)
    pandas2ri.activate()
    df = robjects.conversion.py2rpy(data_trans_pd)
    pval = kpcalg.kernelCItest(**{'suffStat' : rlc.TaggedList((df,"hsic.perm"),tags=('data','ic.method')),
                            'x': A+1,
                            'y': B+1,
                            'S': np.array(list(S))+1})
    return pval

def cits_unrolled(X,tau, alpha= 0.05,cond_dep ='cond_dep_pcorr'):
    """Step in the CITS algorithm to estimation the unrolled causal graph step from time series data

    :param X: pxT array with p variables and T time points
    :type X: numpy.array
    :param tau: Markovian order, in other words, maximum time delay of interaction, i.e. $X_t$ can depend up to $X_{t-\tau}$ and not earlier.
    :type tau: int
    :param alpha: Significance level of conditional dependence tests
        (default is 0.05)
    :type alpha: float
    :param cond_dep: Specifies which conditional dependence test to use. It can take value 'cond_dep_pcorr' for Partial correlation based test suitable for Gaussian noise distribution and 'cond_dep_hsic' for test based on Hilbert-Schmidt criterion applicable for either Gaussian or non-Gaussian noise distributions. Partial correlation based test would be more powerful if the noise distribution is known to be Gaussian.
        (default is 'cond_dep_pcorr')
    :type cond_dep: string
    :returns: Adjacency matrix of unrolled graph
    :rtype: numpy.array
    """ 
        
    if cond_dep == 'cond_dep_hsic':
        cond_dep_test = cond_dep_hsic
    else:
        cond_dep_test = cond_dep_pcorr

    p = X.shape[0]
    n = X.shape[1]
    A = np.ones((p*2*(tau+1),p*2*(tau+1)))- np.diag(np.diag(np.ones((p*2*(tau+1),p*2*(tau+1)))))

    for v in range(p):
        for t in range(2*(tau+1)):
            for v1 in range(p):
                for t1 in range(t,2*(tau+1)):#for not instantaneous use range(t,2*(tau+1)). for instantaneous usde range(t+1,2*(tau+1)).
                    A[t1*p+v1,t*p+v] = 0
    
    chi = data_transform(X,tau)

    t = 2*tau+1
    for v in range(p):
        #print('v '+ str(v))
        for v1 in range(p):
            #print('v1 '+str(v1))
            #for t1 in range(tau+1,2*(tau+1)):#as not instantaneous, no point in considering t1=t
            for t1 in range(tau+1,2*tau+1):
                #print('t1 '+ str(t1))
                #for k in powerset(range(2*(tau+1))):
                for k in powerset(range(2*(tau+1))):
                    if cond_dep_test(chi,(v,t),(v1,t1),k,p, alpha) == 0:
                        A[t1*p+v1,t*p+v] = 0
                        break

    return A

def cits_rolled(A,p,tau):
    """Step in the CITS algorithm to transform the unrolled causal graph to rolled graph
    
    :param p: # of variables
    :type p: int
    :param A: Adjacency matrix of unrolled graph
    :type A: numpy.array
    :param tau: Markovian order, in other words, maximum time delay of interaction, i.e. $X_t$ can depend up to $X_{t-\tau}$ and not earlier.
    :type tau: int
    :returns: Adjacency matrix of rolled graph
    :rtype: numpy.array
    """
    B = np.zeros((p,p))
    t = 2*tau+1
    for v1 in range(p):
        for v2 in range(p):
            for t1 in range(t-tau,t):
                if A[t1*p+v1,t*p+v2] != 0:
                    B[v1,v2]=A[t1*p+v1,t*p+v2]###earlier it was 1 here, this should be same
    return B

def cits_weighted_rolled(A,p,tau):
    """Step in the CITS algorithm to transform the weighted unrolled causal graph to the weighted rolled graph 

    :param A: Adjacency matrix of weighted unrolled graph
    :type A: numpy.array    
    :param p: # of variables
    :type p: int
    :param tau: Markovian order
    :type tau: int
    :returns: Adjacency matrix of weighted rolled graph
    :rtype: numpy.array
    """
    
    B = np.zeros((p,p))
    t = 2*tau+1
    for v1 in range(p):
        for v2 in range(p):
            s = 0
            for t1 in range(t-tau,t):
#                if A[t1*p+v1,t*p+v2] != 0:
                if A[(tau+1)*v1+(t1-t+tau),(tau+1)*v2+tau] !=0:
                    B[v1,v2]+= A[(tau+1)*v1+(t1-t+tau),(tau+1)*v2+tau]#A[t1*p+v1,t*p+v2]
                    s+=1
            if s>0:
                B[v1,v2] = B[v1,v2]/s
    return B

def cits_full(X,tau, alpha=0.05, cond_dep = 'cond_dep_pcorr'):
    """Convenient wrapper for CITS algorithm in computing the unweighted rolled graph from time series data. See cits_full_weighted to compute the weighted rolled graph.
    
    :param X: pxT array for time series with p variables and T time points
    :type X: nump.array
    :param tau: Markovian order, in other words, maximum time delay of interaction, i.e. $X_t$ can depend up to $X_{t-\tau}$ and not earlier.
    :type tau: int
    :param alpha: Significance level of conditional dependence tests
        (default is 0.05)
    :type alpha: float
    :param cond_dep: Specifies which conditional dependence test to use. It can take value 'cond_dep_pcorr' for Partial correlation based test suitable for Gaussian noise distribution and 'cond_dep_hsic' for test based on Hilbert-Schmidt criterion applicable for either Gaussian or non-Gaussian noise distributions. Partial correlation based test would be more powerful if the noise distribution is known to be Gaussian.
        (default is 'cond_dep_pcorr')
    :type cond_dep: string
    
    :cond_dep: choice of conditional dependence test
            :'cond_dep_pcorr': partial correlation based test for Gaussian data
            :'cond_dep_hsic': Hilbert-Schmidt criterion based test for Non-Gaussian data

    :returns: Unweighted adjacency matrix of rolled graph
    :rtype: numpy.array
        (shape is (p,p)) 
    """
    p = X.shape[0]
    A = cits_unrolled(X,tau,alpha, cond_dep)
    B = cits_rolled(A,p,tau)
    return B

def causaleff_lscm(g,data):
    """Assigns weights to edges in the unrolled graph estimate of CITS algorithm

    :param g: Unrolled causal graph
    :type g: networkx.DiGraph 
    :param data: Transformed original time series X by data_transformed function
    :type data: nump.array
    :returns: Adjacency matrix with (i,j) entry having the weight of causal relationship from the variable with index i -> the variable with index j in the unrolled graph g
    :rtype: numpy.array
    """

    Edges = list(g.edges)
    Nodes = list(g.nodes)
    causaleff=np.zeros((len(Nodes),len(Nodes)))

    for x in Nodes:
        for y in Nodes:
            if (x,y) in Edges:
                pa_y = list(g.predecessors(y))
                regressors = pa_y
                X=np.asarray(data[:,regressors])
                Y=np.asarray(data[:,y])
                X0=np.hstack((np.ones((X.shape[0],1)),X))
                lm_out = np.linalg.lstsq(X0,Y,rcond=None)[0]
                causaleff[x,y] = lm_out[regressors.index(x)+1]
    return causaleff


def cits_full_weighted(X,tau, alpha=0.05, cond_dep = 'cond_dep_pcorr', thresh = 10):
    """Convenient wrapper for CITS algorithm in computing the weighted rolled graph from time series data. 
    
    Instead, if you like to compute the unweighted rolled graph, see cits_full. Or alternatively, process the output of this current function to have non-zero edge weights indicating an edge and zero edge weights indicating no edge in the unweighted rolled graph.
    
    :param X: pxT array for time series with p variables and T time points
    :type X: nump.array
    :param tau: Markovian order, in other words, maximum time delay of interaction, i.e. $X_t$ can depend up to $X_{t-\tau}$ and not earlier.
    :type tau: int
    :param alpha: Significance level of conditional dependence tests
        (default is 0.05)
    :type alpha: float
    :param cond_dep: Specifies which conditional dependence test to use. It can take value 'cond_dep_pcorr' for Partial correlation based test suitable for Gaussian noise distribution and 'cond_dep_hsic' for test based on Hilbert-Schmidt criterion applicable for either Gaussian or non-Gaussian noise distributions. Partial correlation based test would be more powerful if the noise distribution is known to be Gaussian.
        (default is 'cond_dep_pcorr')
    :type cond_dep: string
    
    :cond_dep: choice of conditional dependence test
            :'cond_dep_pcorr': partial correlation based test for Gaussian data
            :'cond_dep_hsic': Hilbert-Schmidt criterion based test for Non-Gaussian data

    :returns: Weighted adjacency matrix of rolled graph, whose (i,j) entry represents the weight of causal relationships from variable i -> variable j. A non-zero weight indicates the presence of a causal relationship and zero weight indicates its absence.
    :rtype: numpy.array
        (shape is (p,p))
    """

    import networkx as nx
    p = X.shape[0]
    A = cits_unrolled(X,tau,alpha, cond_dep)
    # g = nx.from_numpy_array(A, create_using= nx.DiGraph())
    B = cits_rolled(A,p,tau)

    U = np.zeros((p*(tau+1),p*(tau+1)))
    t = 2*tau+1
    for v1 in range(p):
        for v2 in range(p):
            for t1 in range(t-tau,t):
                if A[t1*p+v1,t*p+v2] != 0:
                    U[(tau+1)*v1+(t1-t+tau),(tau+1)*v2+tau] = 1
    # for idx1 in range(p):
    #     for idx2 in range(p):
    #         if B[idx1,idx2] == 1:

    g = nx.from_numpy_array(U, create_using= nx.DiGraph())
    data_trans = data_transformed(X, tau)
    causaleff_A = causaleff_lscm(g,data_trans)
    causaleff_B = cits_weighted_rolled(causaleff_A,p,tau)
    causaleff_B[np.abs(causaleff_B)<np.max(causaleff_B)/thresh] = 0
    B_out = (causaleff_B!=0).astype(int)
    return B_out, causaleff_B