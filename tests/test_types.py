from pycharge import Charge


def test_types_imports():
    def f(t):
        return [0.0, 0.0, 0.0]

    assert isinstance(Charge(f), Charge)
    assert callable(f)
