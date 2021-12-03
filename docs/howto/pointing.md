# Calculate the Poynting Vector

The Poynting vector, which is the directional flux of the electromagnetic fields, defined by 
\begin{equation}
    \mathbf{S}=\frac{1}{\mu_0}\mathbf{E}\times\mathbf{B},
\end{equation}
can be calculated in PyCharge along a grid by simply calculating $\mathbf{E}$ and $\mathbf{B}$ and then performing the cross product. An example is shown in Figure 5 in the PyCharge paper, and the corresponding script is given in  [examples/paper_figures/figure5.py](https://github.com/MatthewFilipovich/pycharge/blob/master/examples/paper_figures/figure5.py):

<p align="center">
  <img width="400" src="../../figs/figure5.png">
</p>