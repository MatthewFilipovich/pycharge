"""This module defines the Charge class."""

from typing import NamedTuple

from scipy.constants import e

from pycharge.types import PositionFn


class Charge(NamedTuple):
    position: PositionFn
    q: float = e
