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



PyCharge is an open-source Python library for differentiable electromagnetics simulations of moving point charges with JAX.

Key Features
------------

.. warning::

   TODO!!!
.. .. grid:: 1 2 2 3
..    :gutter: 2

..    .. grid-item-card:: 🌊 **Differentiable Wave Optics**
..       :class-card: sd-bg-light sd-border sd-shadow

..       Model, analyze, and optimize optical systems using Fourier optics.

..    .. grid-item-card:: 🔥 **Built on PyTorch**
..       :class-card: sd-bg-light sd-border sd-shadow

..       GPU acceleration, batch processing, and automatic differentiation.

..    .. grid-item-card:: 🛠️ **End-to-End Optimization**
..       :class-card: sd-bg-light sd-border sd-shadow

..       Joint optimization of optical hardware and machine learning models.

..    .. grid-item-card:: 🔬 **Optical Elements**
..       :class-card: sd-bg-light sd-border sd-shadow

..       Lenses, modulators, detectors, polarizers, and more.

..    .. grid-item-card:: 🖼️ **Spatial Profiles**
..       :class-card: sd-bg-light sd-border sd-shadow

..       Hermite-Gaussian, Laguerre-Gaussian, Zernike modes, and others.

..    .. grid-item-card:: 🔆 **Polarization & Coherence**
..       :class-card: sd-bg-light sd-border sd-shadow

..       Simulate polarized light and fields with arbitrary spatial coherence.

.. _installation:

Installation
------------

PyCharge is available on `PyPI <https://pypi.org/project/pycharge>`_ and can be installed with:

.. code-block:: bash

    pip install pycharge


Contributing
--------------

We welcome contributions! See our `Contributing Guide <https://github.com/MatthewFilipovich/pycharge/blob/main/CONTRIBUTING.md>`_ for details.

Citing PyCharge
-------------------

If you use PyCharge in your research, please cite our `paper <https://doi.org/10.1016/j.cpc.2022.108291>`_:

.. code-block:: bibtex
   @misc{filipovich2022pycharge,
     title={PyCharge: An open-source Python package for self-consistent electrodynamics simulations of Lorentz oscillators and moving point charges},
     author={Matthew J. Filipovich and S. Hughes},
     year={2022},
     journaltitle = {Computer Physics Communications},
     volume = {274},
     pages = {108291},
     issn = {00104655},
     doi = {10.1016/j.cpc.2022.108291},
     url={https://doi.org/10.1016/j.cpc.2022.108291},
   }

License
-------

PyCharge is distributed under the MIT License. See the `LICENSE <https://github.com/MatthewFilipovich/pycharge/blob/main/LICENSE>`_ file for more details.