from pycharge.charge import Charge, SolverConfig


def test_charge_init_custom_values():
    """Test Charge initialization with custom solver config."""

    def custom_position(t):
        return [1.0, 2.0, 3.0]

    config = SolverConfig(
        fixed_point_rtol=1e-6,
        fixed_point_atol=1e-7,
        fixed_point_max_steps=100,
        fixed_point_throw=True,
        root_find_rtol=1e-8,
        root_find_atol=1e-9,
        root_find_max_steps=50,
        root_find_throw=False,
    )

    charge = Charge(
        position_fn=custom_position,
        q=1.0,
        solver_config=config,
    )

    assert charge.position_fn == custom_position
    assert charge.q == 1.0
    assert charge.solver_config.fixed_point_rtol == 1e-6
    assert charge.solver_config.fixed_point_atol == 1e-7
    assert charge.solver_config.fixed_point_max_steps == 100
    assert charge.solver_config.fixed_point_throw is True
    assert charge.solver_config.root_find_rtol == 1e-8
    assert charge.solver_config.root_find_atol == 1e-9
    assert charge.solver_config.root_find_max_steps == 50
    assert charge.solver_config.root_find_throw is False
