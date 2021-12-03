# Simulate Moving Dipoles

In addition to stationary dipoles, PyCharge can self-consistently simulate moving dipoles with a time-dependent origin (center of mass) position.
Other direct EM simulation approaches (e.g., the FDTD method) cannot accurately model moving dipoles, which can have practical importance for nano-scale interactions as real atoms are rarely stationary. Thus, PyCharge can be used to explore new physics phenomena that arise from this additional dipole motion (e.g., phonons in dipole chains).
Simulations with moving dipoles are performed in PyCharge by creating a function that accepts the time $t$ as a parameter and returns the position of the dipole's origin position at $t$ as a three element array ($x$, $y$, $z$). This function is then passed as a parameter when instantiating the `Dipole` object. An example of instantiating a `Dipole` object with a time-dependent origin is given below:

```python
from numpy import pi, cos
import pycharge as pc
def fun_origin(t):
    x = 1e-10*cos(1e12*2*pi*t)
    return ((x, 0, 0))
omega_0 = 100e12*2*pi
init_d = (0, 1e-9, 0)
source = pc.Dipole(omega_0, fun_origin, init_d)
```
