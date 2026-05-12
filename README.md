# Python Package for CITS algorithm: Causal Inference from Time Series data

CITS algorithm infers causal relationships in time series data based on structural causal model and Markovian condition of arbitrary but finite order. See the [paper](https://arxiv.org/abs/2508.01920) for details.

## Installation

You can get the latest version of CITS package as follows

`pip install cits`

## Requirements

- Python >= 3.6
- R >= 4.0
- R package `kpcalg` and its dependencies. They can be installed in R or RStudio as follows:

```
> install.packages("BiocManager")
> BiocManager::install("graph")
> BiocManager::install("RBGL")
> install.packages("pcalg")
> install.packages("kpcalg")
```


## Documentation

[Documentation is available at readthedocs.org](https://cits.readthedocs.io/en/latest/)

## Tutorial

Visit this [Google Colab](https://colab.research.google.com/drive/1TS_uVnbiW9Pb1ywBVjHdL-lnrdFkJ3wp?usp=sharing) for getting started with this package.

Alternatively, see the [Getting Started](https://cits.readthedocs.io/en/latest/gettingstarted.html) in the documentation. 

## Contributing

Your help is absolutely welcome! Please do reach out or create a future branch!

## Citation

Biswas, R., Sripada, S., Mukherjee, S. & Abbasi-Asl, R. (2025) CITS: Nonparametric Statistical Causal Modeling for High-Resolution Neural Time Series. In Review. [https://arxiv.org/abs/2508.01920](https://arxiv.org/abs/2508.01920)
