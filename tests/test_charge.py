from pycharge.charge import Charge


def test_charge_init_custom_values():
    """Test Charge initialization with custom values."""

    def custom_position(t):
        return [1.0, 2.0, 3.0]

    charge = Charge(
        position=custom_position,
        q=1.0,
        fixed_point_rtol=1e-6,
        fixed_point_atol=1e-7,
        fixed_point_max_steps=100,
        fixed_point_throw=True,
        root_find_rtol=1e-8,
        root_find_atol=1e-9,
        root_find_max_steps=50,
        root_find_throw=False,
    )

    assert charge.position == custom_position
    assert charge.q == 1.0
    assert charge.fixed_point_rtol == 1e-6
    assert charge.fixed_point_atol == 1e-7
    assert charge.fixed_point_max_steps == 100
    assert charge.fixed_point_throw is True
    assert charge.root_find_rtol == 1e-8
    assert charge.root_find_atol == 1e-9
    assert charge.root_find_max_steps == 50
    assert charge.root_find_throw is False
