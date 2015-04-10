# Modeling spike-timing dependent plasticity under somato-dendritic prediction error learning

Somato-dendritic prediction error learning was proposed by Urbanczik and Walter in 2014 [1] as a learning rule in a compartmental model that is derived from an error-minimization procedure and can thus be proven to lead to optimal synaptic weight dynamics in a mathematical sense. In [1] it was shown that this learning rule can subserve various learning paradigms depending on how synaptic input is structured. 

The intention of this repository is two-fold:
* provide an implementation of an extended version of the model proposed in [1]. Key changes include
  * conductance-based synapses instead of liner EPSP summation on the dendrite
  * subthreshold voltage flow from the soma to the dendrite
* provide credibility to the proposed learning scheme by showing that a diverse set of characteristics regarding the spike-timing dependence of plasticity observed in experiments emerge under somato-dendritic prediction error learning

The model is implemented in pure Python and relies on the standard software stack for scientific computing in Python (numpy, matplotlib, etc.). I recommend using the [Anaconda distribution](https://store.continuum.io/cshop/anaconda/).

## File description
Here we describe the files containing the main Python code during a simulation:
* model.py contains the key model logic, in particular the rhs of the set of differential equations
* simulation.py contains the main simulation loop which performs Euler integration
* util.py contains helper functions related to the model logic itself, e.g. functions that determine when to initiate spikes
* helper.py contains helper functions unrelated to the model, e.g. saving utilities, IPython notebook creation etc.
* parallelization.py contains a super simple routine for distributing simulation runs across cores (only to be used for "embarrassing parallelization" where the individual runs are completely independent)

## STDP experiments
For every experiment we reproduce, there are three files: 
* a .py file that can be run to perform the simulation and contains more information about the experiment in question, including references
* a .ipynb: an IPython notebook that can be used, once the .py file is run, to analyze the outcome of the simulation
* a .pdf file that contains the resulting figure and is always created in the IPython notebook (see above)
The experiments we recreate in our model are the following:
* stdp_figure_bi_poo: Basic STDP curve as reported by Bi & Poo 1998
* stdp_figure_sjostrom: Frequency dependence of STDP reported by Sjostrom, Turrigian and Nelson 2001
* stdp_figure_artola: Postsynaptic depolarization dependence reported by Ngezahayo, Schachner and Artola 2000
* stdp_figure_sjostrom_switch: Effects of detrimental action potential backpropagation reported by Sjostrom and Hausser 2006
