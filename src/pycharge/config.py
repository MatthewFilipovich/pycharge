from typing import Literal, NamedTuple


class Config(NamedTuple):
    field_component: Literal["total", "velocity", "acceleration"] = "total"
    newton_rtol: float = 0.0
    newton_atol: float = 1.48e-8
    root_find_max_steps: int = 256
