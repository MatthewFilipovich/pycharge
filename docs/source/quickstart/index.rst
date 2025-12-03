Quickstart 
==========

Welcome to PyCharge! This guide provides a hands-on introduction to the core
features of this electromagnetics simulation library.

PyCharge has two primary workflows:

1.  **Point Charge Electromagnetics**: Compute relativistically-correct
    electromagnetic potentials and fields from point charges with predefined
    trajectories.
2.  **Self-Consistent N-Body Electrodynamics**: Simulate the dynamics of
    electromagnetic sources—such as dipoles—that interact through their
    self-generated fields.

This guide will walk you through both. For detailed physical theory, please see
the :doc:`/user-guide/index`.


Installation
------------

Before starting, make sure PyCharge is installed:

.. code-block:: bash

    pip install pycharge

Part 1: Point Charge Electrodynamics
------------------------------------

This workflow is for when you know the trajectory of a charge and want to
calculate the fields it produces. The primary function for this is
``potentials_and_fields``. It takes a list of ``Charge`` objects and returns a
new, highly-optimized function that you can call with spacetime coordinates.

Let's see it in action.

1. Import necessary libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We import PyCharge's core components, JAX for numerical operations, and
Matplotlib for plotting.

.. plot::
    :context: reset

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from scipy.constants import c, e, m_e

    from pycharge import Charge, dipole_source, potentials_and_fields, simulate

    jax.config.update("jax_enable_x64", True)

2. Define a Charge's Trajectory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A charge's trajectory is defined by a simple Python function that takes time
``t`` and returns the charge's ``[x, y, z]`` position.

A key feature of PyCharge is that it leverages JAX's automatic
differentiation (``jax.jacobian``) to automatically calculate the velocity and
acceleration from this position function. You only need to define the path!

Here, we define a charge moving in a circle in the x-y plane.

.. plot::
   :context:

   circular_radius = 1e-10
   velocity = 0.9 * c
   omega = velocity / circular_radius


   def circular_position(t):
       x = circular_radius * jnp.cos(omega * t)
       y = circular_radius * jnp.sin(omega * t)
       z = 0.0
       return x, y, z


   moving_charge = Charge(circular_position, e)

3. Create the Calculation Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We pass a list containing our charge to ``potentials_and_fields``. This
returns a new function that is ready for JAX's just-in-time (JIT)
compilation, making it extremely fast for repeated calculations.

.. plot::
    :context:

    quantities_fn = potentials_and_fields([moving_charge])

    jit_quantities_fn = jax.jit(quantities_fn)

4. Define an Observation Grid and Calculate Quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We'll define a 2D grid in the x-y plane to observe the fields at time ``t = 0``.
The grid points are passed as JAX arrays to our JIT-compiled function.

.. plot::
    :context:

    grid_size = 1000
    xy_max = 5e-9
    x_grid = jnp.linspace(-xy_max, xy_max, grid_size)
    y_grid = jnp.linspace(-xy_max, xy_max, grid_size)
    z_grid = jnp.array([0.0])
    t_grid = jnp.array([0.0])

    X, Y, Z, T = jnp.meshgrid(x_grid, y_grid, z_grid, t_grid, indexing="ij")

    quantities = jit_quantities_fn(X, Y, Z, T)

5. Visualize the Results
~~~~~~~~~~~~~~~~~~~~~~~~

The output ``quantities`` is a ``NamedTuple`` containing JAX arrays for the
scalar potential, vector potential, electric field, and magnetic field.

Let's plot the scalar potential and the magnitude of the electric field.

.. plot::
   :context: close-figs

   scalar_potential = quantities.scalar
   electric_field = quantities.electric
   electric_field_magnitude = jnp.linalg.norm(electric_field, axis=-1)

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

   im1 = ax1.imshow(
       jnp.log10(scalar_potential.squeeze().T),
       extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
       origin="lower",
       cmap="viridis",
       vmax=1,
       vmin=-1,
   )
   fig.colorbar(im1, ax=ax1, label="Scalar Potential (V)")
   ax1.set_xlabel("X Position (m)")
   ax1.set_ylabel("Y Position (m)")
   ax1.set_title("Scalar Potential of a Circularly Moving Charge")

   im2 = ax2.imshow(
       electric_field_magnitude.squeeze().T,
       extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
       origin="lower",
       cmap="inferno",
       vmax=1e10,
       vmin=0,
   )
   fig.colorbar(im2, ax=ax2, label="Electric Field Magnitude (V/m)")
   ax2.set_xlabel("X Position (m)")
   ax2.set_ylabel("Y Position (m)")
   ax2.set_title("Electric Field of a Circularly Moving Charge")

   plt.tight_layout()
   plt.show()

---

Part 2: Self-Consistent N-Body Electrodynamics
----------------------------------------------

PyCharge can also simulate the dynamics of sources whose motion is governed
by the electromagnetic fields they and other sources generate. This is done
using the ``simulate`` function, which takes a sequence of ``Source`` objects
and solves the underlying ordinary differential equations (ODEs).

Here, we simulate a dipole modeled as a **Lorentz oscillator**, a classical
analog for a two-level quantum system.

1. Create a Dipole Source
~~~~~~~~~~~~~~~~~~~~~~~~~

We use the ``dipole_source`` factory to create a dipole. This object bundles
the initial state of the charges with the ODE that governs its motion. We
define its initial charge separation, natural frequency, and other physical
properties.

.. plot::
   :context: reset

   import jax
   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   from scipy.constants import e, m_e

   from pycharge import dipole_source, simulate

   jax.config.update("jax_enable_x64", True)

   dipole = dipole_source(
       d_0=[0.0, 0.0, 1e-9],
       q=e,
       omega_0=100e12 * 2 * jnp.pi,
       m=m_e,
       origin=[0.0, 0.0, 0.0],
   )

2. Set Up and Run the Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We define the time steps for the simulation and then create the simulation
function by passing a list of our sources to ``simulate``.

.. plot::
    :context:

    t_start = 0.0
    t_num = 40_000
    dt = 1e-18
    ts = jnp.linspace(t_start, (t_num - 1) * dt, t_num)

    sim_fn = jax.jit(simulate([dipole], ts))

    source_states = sim_fn()

3. Analyze the Simulation Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``source_states`` contain the position and velocity of each charge in the
source at every time step. Let's plot the z-position of the two charges
in our dipole.

The plot shows a damped oscillation. The dipole loses energy over time because,
as an accelerating source, it radiates electromagnetic waves. This effect,
known as **radiation damping**, is automatically captured by the simulation.

.. plot::
    :context:

    dipole_state = source_states[0]

    position_history = dipole_state[:, :, 0, :]

    charge1_pos = position_history[:, 0, :]
    charge2_pos = position_history[:, 1, :]

    plt.figure(figsize=(10, 6))
    plt.plot(ts, charge1_pos[:, 2], label="Charge 1 (negative)")
    plt.plot(ts, charge2_pos[:, 2], label="Charge 2 (positive)")
    plt.xlabel("Time (s)")
    plt.ylabel("Z Position (m)")
    plt.title("Damped Oscillation of Charges in a Simulated Lorentz Dipole")
    plt.legend()
    plt.grid(True)
    plt.show()

Next Steps
==========

This quickstart has demonstrated the two main workflows in PyCharge:

1.  Calculating fields from charges with predefined trajectories.
2.  Simulating the dynamics of sources interacting with their own fields.

To dive deeper, explore the :doc:`/user-guide/index` for more detailed
explanations of the physics and the :doc:`/examples/index` for more
advanced use cases!