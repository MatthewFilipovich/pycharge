from pycharge import Charge


def test_charge_initialization():
    charge = Charge(lambda t: [0.0, 0.0, 0.0])
    assert charge is not None
