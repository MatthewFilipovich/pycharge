"""PyCharge is a package for executing electrodynamics simulations."""

from .charges import *
from .dipole import Dipole
from .dipole_analyses import *
from .simulation import *


def cite():
    """Print BibTeX citation for the `pycharge` package."""
    citation = """@article{pycharge,
      title={{P}y{C}harge: An open-source Python package for self-consistent electrodynamics simulations of Lorentz oscillators and moving point charges}, 
      author={Matthew J. Filipovich and Stephen Hughes},
      year={2021},
      eprint={2107.12437},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph}
}"""
    print(citation)
