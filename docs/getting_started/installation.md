# Installation

## Requirements

PyCharge and its required dependencies can be installed using [pip](https://pip.pypa.io/en/stable/):

```sh
pip install pycharge
```

To install PyCharge in development mode, clone the GitHub repository and install with pip using the editable option:

```sh
git clone https://github.com/MatthewFilipovich/pycharge
pip install -e ./pycharge
```

## MPI Implementation

PyCharge supports the ability to perform parallel computations across multiple processes using the `run_mpi` method from the `Simulation` class (see [API Reference](../api_reference/simulation-reference.md)). PyCharge uses the python package [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/) to provide bindings of the MPI standard for use in Python.

To use the MPI functionality in PyCharge, **a working MPI implementation must be installed**. A Linux installation guide for MPI implementations can be found [here](https://mpi4py.readthedocs.io/en/stable/appendix.html#building-mpi), and a Windows MPI implementation developed by Microsoft can be downloaded [here](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi).

Once an MPI implementation is installed, the dependent Python package can be installed using pip:

```sh
pip install mpi4py
```
