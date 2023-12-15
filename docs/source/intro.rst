About CITS
==========

``cits`` is a Python package that implements the **Causal Inference in Time Series** (CITS) algorithm for inferring causal relationships (and their strengths) betweem variables from time series data. It uses the non-parametric approach of structural causal modeling that does not assume specific dynamical equation for the time series and uses a Markovian condition of arbitrary but finite order on the time series.  

An application of interest is in neural connectomics to estimate the causal functional connectivity between neurons from neural activity time series.

- **Causal functional connectivity**: The representation of the flow of information between neurons in the brain based on their activity is termed the causal functional connectivity (CFC). The CFC is not directly observed and needs to be inferred from neural time series. 

.. image:: Schematic.png
    :align: center
    :width: 500

CITS is widely applicable for time series data wheneve the data is recorded at a finer time granularity than the time lag of causal effects, to ensure no concurrent causal effects. Examples of such datasets in neurosciences include recordings by the popular Neuropixels technology in animal models, where neurons are recorded at 30 KHz sampling rate which yields one sample per 0.03 ms while neural synaptic transmission has a delay of 0.5-1 ms for adjacent neurons and longer for non-adjacent neurons. This ensures the absence of concurrent causal effects.

CITS has mathematical guarantee in estimating the true causal graph under widely applicable conditions on the underlying time series.

CITS is shown to have greater efficacy than the recent Time-Aware PC algorithm when there are no concurrent causal effects. 

.. The package currently supports the following methods:

.. - :ref:`Time-Aware PC Algorithm <Time-Aware PC Algorithm>`
.. - :ref:`PC Algorithm <PC Algorithm>`
.. - :ref:`Granger Causality <Granger Causality>`