# Point Charges

## Fields and Potentials

The charge and current densities of a point charge $q$ at the position $\mathbf{r}_p(t)$ with velocity $\mathbf{v}(t)$ are, respectively,

$$
    \rho\left(\mathbf{r}, t\right) = q \delta\left[ \mathbf{r} - \mathbf{r}_p\left(t\right)\right], \label{eq1}\tag{1}
$$

$$
    \mathbf{J}\left(\mathbf{r}, t \right) = q \mathbf{v}(t) \delta \left[ \mathbf{r} - \mathbf{r}_p\left(t\right)\right]. \label{eq2}\tag{2}
$$

The scalar and vector potentials of a moving point charge in the Lorenz gauge, known as the [Liénard–Wiechert potentials](https://en.wikipedia.org/wiki/Li%C3%A9nard%E2%80%93Wiechert_potential), are derived from the Maxwell Equations as

$$
    \Phi(\mathbf{r}, t)=\frac{1}{4 \pi \epsilon_{0}} \frac{q c}{(Rc-\mathbf{R} \cdot \mathbf{v})}, \label{eq3}\tag{3}
$$

$$
    \mathbf{A}(\mathbf{r}, t)=\frac{\mathbf{v}}{c^{2}} \Phi(\mathbf{r}, t), \label{eq4}\tag{4}
$$

where $R=|\mathbf{r}-\mathbf{r}_p(t')|$ is the retarded position $\mathbf{r}_p(t')$ to the field point $\mathbf{r}$, and $\mathbf{v}$ is also evaluated at the retarded time $t'$ given by

$$
    t' = t-\frac{R(t')}{c}. \label{eq5}\tag{5}
$$

The physical (gauge-invariant) electric and magnetic fields generated from a moving point charge can be obtained using various approaches, including deriving them directly from their scalar and vector potentials:

$$
    \mathbf{E}(\mathbf{r}, t)=\frac{q}{4 \pi \epsilon_{0}} \frac{R}{(\boldsymbol{R} \cdot \mathbf{u})^{3}}\left[\left(c^{2}-v^{2}\right) \mathbf{u}+\boldsymbol{R} \times(\mathbf{u} \times \mathbf{a})\right], \label{eq6}\tag{6}
$$

$$
    \mathbf{B}(\mathbf{r}, t)=\frac{1}{c} \left[\frac{\mathbf{R}}{R} \times \mathbf{E}\left(\mathbf{r}, t \right) \right], \label{eq7}\tag{7}
$$

where $\mathbf{R}=\mathbf{r}-\mathbf{r}_p(t')$ and $\mathbf{u}= c \mathbf{R}/R-\mathbf{v}$. The first term in Eq. \ref{eq6} is known as the electric Coulomb field and is independent of acceleration, while the second term is known as the electric radiation field and is linearly dependent on $\mathbf{a}$:

$$
    \mathbf{E}_{\mathrm{Coul}}\left(\mathbf{r}, t\right) = \frac{q}{4 \pi \epsilon_{0}} \frac{R}{(\boldsymbol{R} \cdot \mathbf{u})^{3}}\left[ \left(c^{2}-v^{2}\right) \mathbf{u} \right], \label{eq8}\tag{8}
$$

$$
    \mathbf{E}_{\mathrm{rad}}\left(\mathbf{r}, t\right) = \frac{q}{4\pi\epsilon_0} \frac{R}{(\boldsymbol{R} \cdot \mathbf{u})^{3}} \left[ \boldsymbol{R} \times\left(\mathbf{u} \times \mathbf{a}\right) \right]. \label{eq9}\tag{9}
$$

The magnetic Coulomb and radiation field terms can be determined by substituting Eqs. \ref{eq8} and \ref{eq9} into Eq. \ref{eq7}. Notably, the Coulomb field falls off as $1/R^{2}$, similar to the static field, while the radiation field decreases as $1/R$.

## PyCharge Implementation

To computationally solve the above equations for the fields and potentials, PyCharge determines the retarded time $t'$ of the moving point charge at the specified position and time using the secant method (from [`scipy.optimize.newton`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html)) to calculate the approximate solution of Eq. \eqref{eq5} for $t'$. PyCharge supports an arbitrary number of point charges in the simulation by exploiting the superposition principle: the total fields and potentials are determined by calculating the individual contributions from each point charge and then summing the results.

In PyCharge, there are two supported types of charges: they either have predefined trajectories as functions of time in 3D space, or they behave as Lorentz oscillators and their trajectory is determined dynamically at each time step during the simulation. The first type of charge is defined by the [`Charge`](../api_reference/charges-reference.md) class, while the other is defined by the [`Dipole`](../api_reference/dipole-reference.md) class. More information about the Lorentz oscillators and their implementaton in PyCharge is given in the [next section](lorentz_oscillators.md).

A continuous charge density $\rho$ can be approximated in the simulation using numerous point charges within the volume, where the charge value of each point charge depends on $\rho$. Similarly, a continuous current density, described by $\mathbf{J}=\rho \mathbf{v}$, can be approximated in the simulation using evenly spaced point charges traveling along a path where the charge value of each point charge depends on $\mathbf{J}$. The accuracy of the calculated fields and potentials generated by these approximated continuous densities is dependent on both the number of point charges used in the simulation and the distance at the field point from the point charges.

[^1]: J. Jackson. _Classic Electrodynamics_. Chapter 14.1
[^2]: D. Griffiths. _Introduction to Electrodynamics_. Chapter 10.3
