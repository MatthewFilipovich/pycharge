# Create Custom Charges

Custom charge trajectories can be simulated in PyCharge by creating custom subclasses of the `Charge` class and defining its motion along the $x$, $y$, and $z$ directions as functions of time. Below is an example of a custom charge class called `ExampleCharge` which oscillates along the $z$ axis with a charge $q$, angular frequency, and amplitude set during instantiation:

```python
from scipy.constants import e
from numpy import cos

class ExampleCharge(Charge):

    def __init__(self, q=e, omega_0=1e12, amplitude=1e-9):
        super().__init__(q)
        self.omega_0 = omega_0
        self.amplitude = amplitude

    def xpos(self, t):
        return 0

    def ypos(self, t):
        return 0

    def zpos(self, t):
        return self.amplitude*cos(self.omega_0*t)
```

Since the equations defining the velocity and acceleration are not defined, they are calculated by PyCharge using discrete derivative approximations. Objects instantiated from the `ExampleCharge` class can now be used as sources in PyCharge simulations.
