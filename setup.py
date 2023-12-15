import setuptools

with open('README.md','r') as fh:
    README = fh.read()

VERSION = "1.0"

setuptools.setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name = 'cits',
    version = VERSION,
    author = 'Rahul Biswas',
    description = 'CITS algorithm for inferring causality from time series data',
    long_description= README,
    long_description_content_type = 'text/markdown',
    install_requires=['numpy','scipy','pandas','rpy2','networkx',],
    url='https://github.com/biswasr/CITS',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
