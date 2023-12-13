import numpy as np
from itertools import chain, combinations
from scipy import stats, linalg
def data_transform(X,tau):
    """
    Transforms data X of shape (p,n) to time-windowed samples $\chi$ of shape $(p*2*(\tau+1),N)$. 
    :X: pxn array with p variables and n time points
    :tau: maximum time delay of interaction, i.e. $X_t$ can depend up to $X_{t-\tau}$ and not earlier.
    """ 
    p = X.shape[0]
    n = X.shape[1]
    N = int((n-2*(tau+1))/(2*(tau+1)))

    chi = np.zeros((p*2*(tau+1),N))

    #transform X to time-windowed samples chi
    for i in range(N):
        chi[:,i] = X[:,2*(tau+1)*i:2*(tau+1)*(i+1)].T.reshape(p*2*(tau+1))
    return chi

def data_transform_gaps(X,tau):
    """
    Transforms data X of shape (p,n) to time-windowed samples $\chi$ of shape $(p*2*(\tau+1),N)$. 
    :X: pxn array with p variables and n time points
    :tau: maximum time delay of interaction, i.e. $X_t$ can depend up to $X_{t-\tau}$ and not earlier.
    """ 
    p = X.shape[0]
    n = X.shape[1]
    N = int((n-2*(tau+1))/(2*(tau+1)))

    chi = np.zeros((p*2*(tau+1),N))

    #transform X to time-windowed samples chi
    for i in range(N):
        chi[:,i] = X[:,2*(tau+1)*i:2*(tau+1)*(i+1)].T.reshape(p*2*(tau+1))
    
    chi2 = chi[:,::2]
    return chi2

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def cond_dep_pcorr(chi,i1,j1,k,p,alpha=0.05):
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
    import pandas as pd
    #from hsiccondTestIC import hsic_CI
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
    # if len(S) == 0:
    #     X=data[:,A]
    #     Y=data[:,B]
    #     pval=hsic_CI(X,Y,reps=reps)
    # else:
    #     p = data.shape[1]
    #     idx = np.zeros(p, dtype=bool)
    #     for i in range(p):
    #         if i in S:
    #             idx[i]=True
    #     X=data[:,A]
    #     Y=data[:,B]
    #     Z=data[:,idx]
    #     pval=hsic_CI(X,Y,Z,reps=reps)
    return pval

def cits_unrolled(X,tau, alpha= 0.05,cond_dep ='cond_dep_pcorr'):
    """
    :X: pxn array with p variables and n time points
    :tau: maximum time delay of interaction, i.e. $X_t$ can depend up to $X_{t-\tau}$ and not earlier.
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
    """
    :p: # of variables
    :A: Unrolled adjacency matrix
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
    """
    :p: # of variables
    :A: Unrolled adjacency matrix
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
    p = X.shape[0]
    A = cits_unrolled(X,tau,alpha, cond_dep)
    B = cits_rolled(A,p,tau)
    return B
def causaleff_ida(g,data):
    #from sklearn import linear_model
    #import statsmodels.api as sm
    import numpy as np
    #from gen_data_fns import sigmoid, relu
    Edges = list(g.edges)
    Nodes = list(g.nodes)
    causaleff=np.zeros((len(Nodes),len(Nodes)))

    for x in Nodes:
        for y in Nodes:
            if (x,y) in Edges:
            #x!=y:# and x in list(nx.ancestors(g,y)):#list(g.predecessors(y)):#list(nx.ancestors(g,y)):
            #if (x,y) in Edges:
            #if y not in list(g.predecessors(x)):
            #if x in list(nx.ancestors(g,y)):
                #lm = linear_model.LinearRegression()
                pa_x = list(g.predecessors(x))
                pa_y = list(g.predecessors(y))
                if x not in pa_x:
                    regressors = pa_x + [x]
                else:
                    regressors = pa_x
                if y in pa_x:
                    causaleff[x,y] = 0
                else:
                    X=np.asarray(data[:,regressors])
                    Y=np.asarray(data[:,y])
                    X0=np.hstack((np.ones((X.shape[0],1)),X))
                    lm_out = np.linalg.lstsq(X0,Y,rcond=None)[0]
                    causaleff[x,y] = lm_out[regressors.index(x)+1]
    return causaleff

def causaleff_lscm(g,data):
    #from sklearn import linear_model
    #import statsmodels.api as sm
    import numpy as np
    #from gen_data_fns import sigmoid, relu
    Edges = list(g.edges)
    Nodes = list(g.nodes)
    causaleff=np.zeros((len(Nodes),len(Nodes)))

    for x in Nodes:
        for y in Nodes:
            if (x,y) in Edges:
            #x!=y:# and x in list(nx.ancestors(g,y)):#list(g.predecessors(y)):#list(nx.ancestors(g,y)):
            #if (x,y) in Edges:
            #if y not in list(g.predecessors(x)):
            #if x in list(nx.ancestors(g,y)):
                #lm = linear_model.LinearRegression()
                #pa_x = list(g.predecessors(x))
                pa_y = list(g.predecessors(y))
                # if x not in pa_x:
                #     regressors = pa_x + [x]
                regressors = pa_y
                # if y in pa_x:
                #     causaleff[x,y] = 0
                # else:
                X=np.asarray(data[:,regressors])
                Y=np.asarray(data[:,y])
                X0=np.hstack((np.ones((X.shape[0],1)),X))
                lm_out = np.linalg.lstsq(X0,Y,rcond=None)[0]
                causaleff[x,y] = lm_out[regressors.index(x)+1]
    return causaleff

def data_transformed(data, lag):
    import numpy as np
    n = data.shape[0]
    p = data.shape[1]
    lag1=lag+1
    new_n = int(np.floor((n-lag)/(2*lag1))*(2*lag1))
    data=data[:new_n,:]
    data2=np.zeros((int(new_n/(2*lag1)),p*lag1))
    for i in range(p):
        for j in range(lag1):
            data2[:,lag1*i+j]=data[j::(2*lag1),i]
    return data2

def cits_full_weighted(X,tau, alpha=0.05, cond_dep = 'cond_dep_pcorr', thresh = 10):
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
    data_trans = data_transformed(X.T, tau)
    causaleff_A = causaleff_lscm(g,data_trans)#causaleff_ida(g,data_trans)
    causaleff_B = cits_weighted_rolled(causaleff_A,p,tau)
    causaleff_B[np.abs(causaleff_B)<np.max(causaleff_B)/thresh] = 0
    B_out = (causaleff_B!=0).astype(int)
    return B_out, causaleff_B

def cits_unrolled_in(X,tau,alpha, cond_dep = 'cond_dep_pcorr'):
    """
    :X: pxn array with p variables and n time points
    :tau: maximum time delay of interaction, i.e. $X_t$ can depend up to $X_{t-\tau}$ and not earlier.
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
                for t1 in range(t+1,2*(tau+1)):#for not instantaneous use range(t,2*(tau+1)). for instantaneous usde range(t+1,2*(tau+1)).
                    A[t1*p+v1,t*p+v] = 0
    
    chi = data_transform(X,tau)

    t = 2*tau+1
    S={}
    for v in range(p):
        S[(t,v)]={}
        for v1 in range(p):
            for t1 in range(tau+1,2*(tau+1)):
                S[(t,v)][(t1,v1)]=set()
                for k in powerset(range(2*(tau+1))):
                    if cond_dep_test(chi,(v,t),(v1,t1),k,p, alpha) == 0:
                        A[t1*p+v1,t*p+v] = 0
                        if t1==t:
                            S[(t,v)][(t1,v1)]=S[(t,v)][(t1,v1)] | set(list(k))
                        break

    A1 = unrolled_in_orient(A,S,tau,p)
    return A1

def unrolled_in_orient(A,S,tau,p):
    t = 2*tau+1
    for v in range(p):
        for v1 in range(p):
            if A[t*p+v1,t*p+v] == 0:
                for c in set(list(range(p)))-{v,v1}:
                    if A[t*p+v1,t*p+c]==1 and A[t*p+c,t*p+v1]==1 and A[t*p+c,t*p+v]==1 and A[t*p+v,t*p+c]==1 and t*p+c not in S[(t,v)][(t,v1)]:
                        A[t*p+c,t*p+v1]=0
                        A[t*p+c,t*p+v]=0
                    if A[t*p+v1,t*p+c]==1 and A[t*p+c,t*p+v1]==0 and A[t*p+c,t*p+v]==1 and A[t*p+v,t*p+c]==1 and t*p+c not in S[(t,v)][(t,v1)]:
                        A[t*p+v,t*p+c]=0
            if A[t*p+v1,t*p+v] == 1 and A[t*p+v,t*p+v1] == 1:
                for c in set(list(range(p)))-{v,v1}:
                    if A[t*p+v1,t*p+c]==1 and A[t*p+c,t*p+v1]==0 and A[t*p+c,t*p+v]==1 and A[t*p+v,t*p+c]==0:
                        A[t*p+v,t*p+v1]==0
                        for d in set(list(range(p)))-{v,v1,c}:
                            if A[t*p+c,t*p+d] == 0 and A[t*p+d,t*p+c] == 0:
                                if A[t*p+v1,t*p+c]==1 and A[t*p+c,t*p+v1]==1 and A[t*p+c,t*p+v]==1 and A[t*p+v,t*p+c]==0 and A[t*p+v1,t*p+d]==1 and A[t*p+d,t*p+v1]==1 and A[t*p+d,t*p+v]==1 and A[t*p+v,t*p+d]==0:
                                    A[t*p+v,t*p+v1] = 0
                                if A[t*p+v1,t*p+c]==1 and A[t*p+c,t*p+v1]==1 and A[t*p+c,t*p+d]==1 and A[t*p+d,t*p+c]==0 and A[t*p+c,t*p+d]==1 and A[t*p+d,t*p+c]==0 and A[t*p+d,t*p+v]==1 and A[t*p+v,t*p+d]==0:
                                    A[t*p+v,t*p+v1] = 0
    return A

def cits_rolled_in(A1,p,tau):
    """
    :p: # of variables
    :A: Unrolled adjacency matrix
    """
    B = np.zeros((p,p))
    t = 2*tau+1
    for v1 in range(p):
        for v2 in range(p):
            for t1 in range(t-tau,t):
                if A1[t1*p+v1,t*p+v2] != 0:
                    B[v1,v2]=1
    return B

def cits_full_in(X,tau, alpha=0.05, cond_dep = 'cond_dep_pcorr'):
    p = X.shape[0]
    A = cits_unrolled_in(X,tau,alpha, cond_dep)
    B = cits_rolled_in(A,p,tau)
    return B