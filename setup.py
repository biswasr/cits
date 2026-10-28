import setuptools

with open('README.md','r') as fh:
    README = fh.read()

import re
with open('cits/__init__.py', 'r') as fh:
    VERSION = re.search(r'^__version__ = "([^"]+)"', fh.read(), re.M).group(1)

setuptools.setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name = 'cits',
    version = VERSION,
    author = 'Rahul Biswas',
    description = 'CITS algorithm for inferring causality from time series data',
    long_description= README,
    long_description_content_type = 'text/markdown',
    # Core deps: base CITS, the CPU skeleton, GPU/contemporaneous wiring, and the
    # weighted graph (networkx) all run on Python + these. rpy2/R is NOT a
    # core requirement: it is needed only for the optional non-Gaussian HSIC
    # conditional-independence test (extras_require['hsic']). cuPC is a
    # compiled CUDA artifact (not on PyPI); see the README GPU setup section.
    install_requires=['numpy','scipy','pandas','networkx',],
    extras_require={
        # Non-Gaussian HSIC CI test via R's kpcalg (also requires an R
        # install with the kpcalg package; see README).
        'hsic': ['rpy2'],
        # Plotting helpers (cits.plot_graph / cits.plot_matrix). networkx is
        # already a core dependency; matplotlib is only needed for plotting.
        'viz': ['matplotlib'],
        # RCIT kernel CI test with the batched gamma null (CPU or CUDA).
        'rcit': ['torch'],
    },
    url='https://github.com/abbasilab/cits',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',  # 'from __future__ import annotations' requires 3.7+
    license='PolyForm-Noncommercial-1.0.0',
    license_files=['LICENSE'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: Other/Proprietary License",
    ],
)
