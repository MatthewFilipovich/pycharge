.. raw:: html

   <style>
   .bd-sidebar-secondary {
       display: none;
   }
   </style>

.. toctree::
   :hidden:

   auto_quickstart/quickstart
   user_guide/index
   auto_examples/index
   reference/index

PyCharge Documentation
==========================

.. warning::

   TODO!!!

PyCharge is an open-source Python library for differentiable electromagnetics simulations of moving point charges with JAX.

Key Features
------------

TODO!

.. .. grid:: 1 2 2 3
..    :gutter: 2

..    .. grid-item-card:: 🌊 **Differentiable Wave Optics**
..       :class-card: sd-bg-light sd-border sd-shadow

..       A comprehensive framework for modeling, analyzing, and designing optical systems using differentiable Fourier optics.

..    .. grid-item-card:: 🔥 **Built on PyTorch**
..       :class-card: sd-bg-light sd-border sd-shadow

..       Leverages PyTorch for GPU acceleration, batch processing, automatic differentiation, and efficient gradient-based optimization.  

..    .. grid-item-card:: 🛠️ **End-to-End Optimization**
..       :class-card: sd-bg-light sd-border sd-shadow

..       Enables optimization of optical hardware and deep learning models within a unified, differentiable pipeline.

..    .. grid-item-card:: 🔬 **Optical Elements**
..       :class-card: sd-bg-light sd-border sd-shadow

..       Features standard optical elements like modulators, lenses, detectors, and polarizers.

..    .. grid-item-card:: 🖼️ **Spatial Profiles**
..       :class-card: sd-bg-light sd-border sd-shadow

..       Provides a wide range of spatial profiles, including Hermite-Gaussian and Laguerre-Gaussian beams.

..    .. grid-item-card:: 🔆 **Polarization & Coherence**
..       :class-card: sd-bg-light sd-border sd-shadow

..       Supports simulations of polarized light and optical fields with arbitrary spatial coherence.

.. _installation:

Installation
------------

To install the latest **stable release** of PyCharge from `PyPI <https://pypi.org/project/pycharge>`_ , run:

.. code-block:: bash

    pip install pycharge

For the latest **development version**, install directly from `GitHub <https://github.com/MatthewFilipovich/pycharge>`_:


.. code-block:: bash

    git clone https://github.com/MatthewFilipovich/pycharge
    cd pycharge
    pip install -e '.[dev]'

This installs the library in editable mode, along with additional dependencies for development and testing.


Contributing
--------------

We welcome bug reports, questions, and feature suggestions to improve PyCharge.

- **Found a bug or have a question?** Please `open an issue on GitHub <https://github.com/MatthewFilipovich/pycharge/issues>`_.
- **Want to contribute a new feature?** Follow these steps:

1. **Fork the repository**: Go to `https://github.com/MatthewFilipovich/pycharge/fork <https://github.com/MatthewFilipovich/pycharge/fork>`_
2. **Create a feature branch**: ``git checkout -b feature/fooBar``
3. **Commit your changes**: ``git commit -am 'Add some fooBar'``
4. **Push to the branch**: ``git push origin feature/fooBar``
5. **Submit a Pull Request**: Open a Pull Request on GitHub

Citing PyCharge
-------------------

If you use PyCharge in your research, please cite our paper:

    M. Filipovich and S. Hughes, *PyCharge: An open-source Python package for self-consistent electrodynamics simulations of Lorentz oscillators and moving point charges*, `Comput. Phys. Commun. <https://doi.org/10.1016/j.cpc.2022.108291>`_ 274, 108291 (2022).
License
-------

PyCharge is distributed under the MIT License. See the `LICENSE <https://github.com/MatthewFilipovich/pycharge/blob/main/LICENSE>`_ file for more details.