# Modeling spike-timing dependent plasticity under somato-dendritic prediction error learning

Somato-dendritic prediction error learning was proposed by [Urbanczik and Senn](http://www.ncbi.nlm.nih.gov/pubmed/24507189) in 2014 as a learning rule in a compartmental model that is derived from an error-minimization procedure and can thus be proven to lead to optimal synaptic weight dynamics in a mathematical sense. It was shown that this learning rule can subserve various learning paradigms depending on how synaptic input is structured. 

The intention of this repository is two-fold:
* provide an implementation of an extended version of the model proposed by Urbanczik and Senn. Key changes include
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
* a .py file that can be run to perform the simulation and will write multiple .p files to disk that contain the simulation results
* an .ipynb file: an IPython notebook that contains analysis code, reading in the associated .p files and producing a figure
* a .pdf file: the figure produced by the IPython notebook

The experiments we recreate in our model are the following:
* stdp_figure_bi_poo: Basic STDP curve as reported by [Bi & Poo 1998](http://www.ncbi.nlm.nih.gov/pubmed/9852584)
* stdp_figure_sjostrom: Frequency dependence of STDP reported by [Sjostrom, Turrigian and Nelson 2001](http://www.ncbi.nlm.nih.gov/pubmed/11754844)
* stdp_figure_artola: Postsynaptic depolarization dependence reported by [Ngezahayo, Schachner and Artola 2000](http://www.ncbi.nlm.nih.gov/pubmed/10729325)
* stdp_figure_sjostrom_switch: Effects of detrimental action potential backpropagation reported by [Sjostrom and Hausser 2006](http://www.ncbi.nlm.nih.gov/pubmed/16846857)

These preliminary data were presented as a [poster](http://dspicher.github.io/pages/dendrites15.html) at the Dendrites conference 2015 in Ventura, California.

This code is released under the [Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license]
(https://creativecommons.org/licenses/by-nc-nd/4.0/)
