"""PyCharge is a package for executing electrodynamics simulations."""

from .charges import *
from .dipole import Dipole
from .dipole_analyses import *
from .simulation import *


def cite():
    """Print BibTeX citation for the `pycharge` package."""
    citation = """@article{pycharge,
    title = {{PyCharge}: {An} open-source {Python} package for self-consistent electrodynamics simulations of {Lorentz} oscillators and moving point charges},
    author = {Filipovich, Matthew J. and Hughes, Stephen},
    journal = {Computer Physics Communications},
    volume = {274},
    pages = {108291},
    year = {2022},
    doi = {10.1016/j.cpc.2022.108291},
    url = {https://doi.org/10.1016/j.cpc.2022.108291},
}"""
    print(citation)
