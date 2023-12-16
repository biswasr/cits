========
Examples
========

Use this tutorial to try out main functionalities of this library and test if it has installed properly. 
In this tutorial, you will be estimating the adjacency matrix of the rolled causal graph from simulated datasets. In neural connectomics, the rolled causal graph between neurons is referred as the causal functional connectivity between neurons. 

Load the CITS package
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

	from cits import methods

Load other packages
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

	import time
	import numpy as np
	from numpy.random import default_rng
	rng = default_rng(seed=111)

Simulate a dataset
~~~~~~~~~~~~~~~~~~

Use the simulate_timeseries function to simulate a time series dataset. It also outputs the adjacency matrix (unweighted and weighted) of the rolled causal graph corresponding to the simulated time series. 

Note that the function inputs the ground truth adjacency matrix desired for the simulated time series, the model type and noise std deviation.

Choices of model type are: 

- model = 'lingauss1' for Linear Gaussian Model 1
- model = 'lingauss2' for Linear Gaussian Model 2
- model = 'nonlinnongauss1' for Non-linear Non-Gaussian Model 1
- model = 'nonlinnongauss2' for Non-linear Non-Gaussian Model 2
- model = 'ctrnn' for CTRNN

The following example uses model = 'lingauss1':

.. code-block:: python

	T=1000
	n_neurons = 4
	noise = 1
	alpha = 0.05
	lag=1

	X, adj, adj_w = simulate_timeseries(model = 'lingauss1', noise = noise, T = T)

Estimate the adjacency matrix of the rolled causal graph of time series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Unweighted adjacency: (i,j) entry represents causal influence from i->j.

Use the cits_full to obtain the unweighted adjacency matrix.

.. code-block:: python

	print("True unweighted adjacency matrix: \n")
	print(adj)
	
	startime = time.time()
	adj_matrix = methods.cits_full(X,lag,alpha)
	out = str(time.time()-startime)
	print("time taken "+ out + "\n")

	print("Estimated unweighted adjacency matrix: \n")
	print(adj_matrix)

- Weighted adjacency: (i,j) entry has strength of causal influence from i->j.

Use the cits_full_weighted to obtain the weighted adjacency matrix.

.. code-block:: python

	print("True weighted adjacency matrix: \n")
	print(adj_w)

	startime = time.time()
	adj_matrix, causaleff = methods.cits_full_weighted(X,lag,alpha)
	out = str(time.time()-startime)
	print("time taken "+ out + "\n")

	print("Estimated weighted adjacency matrix: \n")
	print(causaleff)