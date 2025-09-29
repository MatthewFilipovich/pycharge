from typing import Literal, NamedTuple


class RootFindConfig(NamedTuple):
    rtol: float = 0.0
    atol: float = 1.48e-8
    max_steps: int = 256


class Config(NamedTuple):
    field_component: Literal["total", "velocity", "acceleration"] = "total"
    root_find: RootFindConfig = RootFindConfig()
