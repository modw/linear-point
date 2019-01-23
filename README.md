# Linear Point

Scripts and notebooks to calculate and analyse the BAO linear point for different cosmologies. 


## Python scripts 
`linear_point.py`

Script to calculate the linear point from Cosmological parameters. Basically, pycamb is used to get the linear power spectrum. From the power spectrum, the two point correlation function and its dip and peak are found in the appropriate scale range. The position of these extremes can then be used to obtain the linear point.

`df_splitter.py` and `df_concat.py`

Scripts to split and concatenate pandas dataframes. These can be used to trivially paralellize calculations in a high-performance computing cluster.

`lp_from_df_[chain].py`

Script that utilizes `linear_point.py` to calculate linear point from MCMC chains in pandas dataframe format.

## Jupyter notebooks

`Visualizing Results-paper.ipynb`: Tentative plots for publication. Robust to different chains. Most up to date plotting file. `Visualizing Results.ipynb` is outdated.

`LP RD characterization.ipynb`: Simple comparison of how rdrag and the linear point vary as we change some cosmological parameters.

Every other notebook is outdated and can be ignored.

## Notes

See https://github.com/modw/personal-python/blob/master/scatter_plot_matrix.py for scatter plot matrix plotting function. See [this blog post](https://marcioodwyer.wordpress.com/2019/01/03/my-python-class-to-plot-correlated-variables/) to learn how to use it.

## References

> Anselmi, Stefano, Glenn D. Starkman, and Ravi K. Sheth. "Beating non-linearities: improving the baryon acoustic oscillations with the linear point." Monthly Notices of the Royal Astronomical Society 455.3 (2015): 2474-2483.

> Anselmi, Stefano, et al. "The Linear Point: A cleaner cosmological standard ruler." arXiv preprint arXiv:1703.01275 (2017).

> Anselmi, Stefano, et al. "Linear point standard ruler for galaxy survey data: Validation with mock catalogs." Physical Review D 98.2 (2018): 023527.

> Anselmi, Stefano, et al. "Cosmic distance inference from purely geometric BAO methods: Linear Point standard ruler and Correlation Function Model Fitting." arXiv preprint arXiv:1811.12312 (2018).


