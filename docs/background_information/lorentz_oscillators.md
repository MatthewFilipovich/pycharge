# Lorentz Oscillators

PyCharge can self-consistently model dipoles to account for radiation reaction forces, caused by coupling with the scattered field $\mathbf{E}_s$, by treating the dipoles as Lorentz oscillators. The dipole moment $\boldsymbol{\mu}$ is defined by the differential equation for a damped, driven harmonic oscillator:

$$
    \frac{\mathrm{d}^{2}}{\mathrm{~d} t^{2}} \boldsymbol{\mu}(t)+\Gamma_{0} \frac{\mathrm{d}}{\mathrm{d} t} \boldsymbol{\mu}(t)+\omega_{0}^{2} \boldsymbol{\mu}(t)=\frac{q^{2}}{m} \mathbf{E}_{\mathrm{s}}(t),
$$

where $\Gamma_0$ is the classical expression for the atomic decay rate (caused by radiation reaction and vacuum fluctuations) given by

$$
\Gamma_{0}=q_{\mathrm{i}} \frac{1}{4 \pi \varepsilon_{0}} \frac{2 q^{2} \omega_{0}^{2}}{3 m c^{3}}.$$

The simulation calculates the dipole moment at each time step using the [RK4 method](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods):

<p align="center">
  <img width="600" src="../../figs/algorithm1.jpg">
</p>

There's a lot more to discuss here...

[^1]: L. Novotny and B. Hecht, _Principles of Nano-Optics_, Chapter 8.5