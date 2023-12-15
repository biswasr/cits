Installation
============

Installation with ``pip`` is recommended. For detailed instructions, see below.

Requirements
------------
- Python >=3.6 and R >=4.0
- Install the R package requirements as follows in R.

.. code-block:: R

	> install.packages("BiocManager")
	> BiocManager::install("graph")
	> BiocManager::install("RBGL")
	> install.packages("pcalg")
	> install.packages("kpcalg")

Install ``cits`` using ``pip``
-------------------------------------

.. code-block:: bash

	$ pip install cits