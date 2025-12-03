Point Charges
=============

Fields and Potentials
---------------------

The charge and current densities of a point charge :math:`q` at the position :math:`\mathbf{r}_p(t)` with velocity :math:`\mathbf{v}(t)` are, respectively,

.. math::
   :label: eq1

   \rho\left(\mathbf{r}, t\right) = q \, \delta\!\left[ \mathbf{r} - \mathbf{r}_p\left(t\right)\right]

.. math::
   :label: eq2

   \mathbf{J}\left(\mathbf{r}, t \right) = q \, \mathbf{v}(t) \, \delta \!\left[ \mathbf{r} - \mathbf{r}_p\left(t\right)\right]

The scalar and vector potentials of a moving point charge in the Lorenz gauge, known as the `Liénard–Wiechert potentials <https://en.wikipedia.org/wiki/Li%C3%A9nard%E2%80%93Wiechert_potential>`_, are derived from the Maxwell equations as

.. math::
   :label: eq3

   \Phi(\mathbf{r}, t)=\frac{1}{4 \pi \epsilon_{0}} \frac{q c}{(Rc-\mathbf{R} \cdot \mathbf{v})}

.. math::
   :label: eq4

   \mathbf{A}(\mathbf{r}, t)=\frac{\mathbf{v}}{c^{2}} \, \Phi(\mathbf{r}, t)

where :math:`R = |\mathbf{r}-\mathbf{r}_p(t')|` is the retarded position :math:`\mathbf{r}_p(t')` to the field point :math:`\mathbf{r}`, and :math:`\mathbf{v}` is also evaluated at the retarded time :math:`t'` given by

.. math::
   :label: eq5

   t' = t-\frac{R(t')}{c}

The physical (gauge-invariant) electric and magnetic fields generated from a moving point charge can be obtained using various approaches, including deriving them directly from their scalar and vector potentials:

.. math::
   :label: eq6

   \mathbf{E}(\mathbf{r}, t)=\frac{q}{4 \pi \epsilon_{0}} 
   \frac{R}{(\boldsymbol{R} \cdot \mathbf{u})^{3}}
   \left[\left(c^{2}-v^{2}\right) \mathbf{u}+\boldsymbol{R} \times(\mathbf{u} \times \mathbf{a})\right]

.. math::
   :label: eq7

   \mathbf{B}(\mathbf{r}, t)=\frac{1}{c} 
   \left[\frac{\mathbf{R}}{R} \times \mathbf{E}\left(\mathbf{r}, t \right) \right]

where :math:`\mathbf{R}=\mathbf{r}-\mathbf{r}_p(t')` and :math:`\mathbf{u}= c \mathbf{R}/R-\mathbf{v}`.  
The first term in Eq. :eq:`eq6` is known as the *electric Coulomb field* and is independent of acceleration, while the second term is known as the *electric radiation field* and is linearly dependent on :math:`\mathbf{a}`:

.. math::
   :label: eq8

   \mathbf{E}_{\mathrm{Coul}}\left(\mathbf{r}, t\right) 
   = \frac{q}{4 \pi \epsilon_{0}} 
   \frac{R}{(\boldsymbol{R} \cdot \mathbf{u})^{3}}
   \left[ \left(c^{2}-v^{2}\right) \mathbf{u} \right]

.. math::
   :label: eq9

   \mathbf{E}_{\mathrm{rad}}\left(\mathbf{r}, t\right) 
   = \frac{q}{4\pi\epsilon_0} 
   \frac{R}{(\boldsymbol{R} \cdot \mathbf{u})^{3}} 
   \left[ \boldsymbol{R} \times\left(\mathbf{u} \times \mathbf{a}\right) \right]

The magnetic Coulomb and radiation field terms can be determined by substituting Eqs. :eq:`eq8` and :eq:`eq9` into Eq. :eq:`eq7`.  
Notably, the Coulomb field falls off as :math:`1/R^{2}`, similar to the static field, while the radiation field decreases as :math:`1/R`.

PyCharge Implementation
-----------------------

To computationally solve the above equations, **PyCharge** must determine the retarded time :math:`t'` for each charge at every observation point. This is a root-finding problem for Eq. :eq:`eq5`. PyCharge leverages the `Optimistix <https://docs.kidger.site/optimistix/>`_ library, a JAX-native optimization library, to solve this efficiently. It uses a Newton solver by default, which is fully differentiable and hardware-accelerated.

PyCharge supports an arbitrary number of sources in the simulation by exploiting the superposition principle. The total fields and potentials are determined by calculating the individual contributions from each point charge and then summing the results, all within a single, vectorized JAX computation.

In PyCharge, all sources of fields are defined by their constituent charges. A :class:`~pycharge.Charge` object encapsulates the physical properties of a point charge, most importantly its trajectory.

Defining a :class:`~pycharge.Charge`
~~~~~~~~~~~~~~~~~~~~~

A charge's trajectory is defined with a simple Python function that takes a time ``t`` and returns the ``(x, y, z)`` position. This function is passed to the ``Charge`` object upon creation.

.. code-block:: python

   import jax.numpy as jnp
   from scipy.constants import e
   from pycharge import Charge

   # Define a trajectory for a charge spiraling along the z-axis
   def helical_position(t):
       radius = 1e-10
       omega = 1e16
       velocity_z = 1e6

       x = radius * jnp.cos(omega * t)
       y = radius * jnp.sin(omega * t)
       z = velocity_z * t
       return x, y, z

   # Create a Charge object with this trajectory
   spiraling_charge = Charge(position=helical_position, q=-e)

The velocity and acceleration required for Eq. :eq:`eq6` are derived automatically from the position function using JAX's automatic differentiation capabilities (via ``jax.jacobian``).

PyCharge supports two main ways to define sources:

- **Predefined Trajectories**: For sources like the one above, where the path is known analytically. You define one or more :class:`~pycharge.Charge` objects and pass them to the :func:`~pycharge.potentials_and_fields` function.
- **Dynamic Trajectories**: For sources whose motion is governed by the fields in the system, such as a Lorentz oscillator. These are created using factory functions like :func:`~pycharge.dipole_source`, which generate :class:`~pycharge.Source` objects containing an ODE that is solved by the :func:`~pycharge.simulate` function. See :doc:`lorentz-oscillators` for more details.

A continuous charge density :math:`\rho` can be approximated in the simulation using numerous point charges within a volume, where the charge value of each point charge depends on :math:`\rho`.
Similarly, a continuous current density, described by :math:`\mathbf{J}=\rho \mathbf{v}`, can be approximated using evenly spaced point charges traveling along a path, where the charge value of each point charge depends on :math:`\mathbf{J}`.
The accuracy of the calculated fields and potentials generated by these approximated continuous densities depends on both the number of point charges used and the distance from the field point.

----

.. rubric:: References

[1] J. D. Jackson, *Classical Electrodynamics*, Ch. 14.1  
[2] D. J. Griffiths, *Introduction to Electrodynamics*, Ch. 10.3