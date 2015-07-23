
"""
Here we reproduce experiments reported in
"Synaptic Modifications in Cultured Hippocampal Neurons:
Dependence on Spike Timing, Synaptic Strength, and
Postsynaptic Cell Type"
Guo-qiang Bi and Mu-ming Poo
The Journal of Neuroscience, 1998

Specifically, we investigate the basic spike-timing dependence  of plasticity
by manipulating the relative difference of pre- and postsynaptic spikes
(Figure 7). The data from this figure can be found in
the "experimental_data" folder.

Approximate runtime on an Intel Xeon X3470 machine (4 CPUs, 8 threads):
< 2min

Running this file should produce 101 .p files.

Afterwards, code in the corresponding
IPython notebook will produce a figure showing experimental data and
simulation results next to each other.
"""

from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_phi_spiker, get_dendr_spike_det, get_fixed_spiker
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time
import os

def task((repetition_i,p)):

    n_vary = 5

    values = {True: {"alpha":-55.0,
                     "beta":0.4,
                     "r_max":0.3},
              False: {"alpha":-59.0,
                      "beta":0.5,
                      "r_max":0.17}}

    vary = {"alpha":(-2.0,2.0),
              "beta":(-0.1,0.2),
              "r_max":(-0.05,0.15)}

    down = vary[p["vary"]][0]
    up = vary[p["vary"]][1]
    middle = values[p["h1"]][p["vary"]]
    vary_val = np.linspace(middle+down, middle+up, n_vary)[p["i"]]
    values[p["h1"]][p["vary"]] = vary_val
    values = values[p["h1"]]

    neuron = get_default("neuron")
    neuron["phi"]['r_max'] = values["r_max"]
    neuron["phi"]['alpha'] = values["alpha"]
    neuron["phi"]['beta'] = values["beta"]

    learn = get_default("learn")
    learn["eps"] = learn["eps"]*p["l_f"]
    learn["eta"] = learn["eta"]*p["l_f"]
    if not p["h1"]:
        learn["eta"] = learn["eta"]*2.5
    else:
        learn["eta"] = learn["eta"]*1.3


    spikes = np.array([61.0])

    my_s = {
        'start': 0.0,
        'end': 150.0,
        'dt': 0.05,
        'pre_spikes': [spikes + p["delta"]],
        'I_ext': lambda t: 0.0
        }


    seed = 1
    accs = [PeriodicAccumulator(['y','weights'], interval=10), BooleanAccumulator(['spike', 'dendr_spike', 'pre_spikes'])]
    if p["h1"]:
        accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det(-50.0,10.0), accs, seed=seed, neuron=neuron, learn=learn, h=1.0)
    else:
        accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det(-50.0,10.0), accs, seed=seed, neuron=neuron, learn=learn)

    dump((accums, values),'bi_poo_vary/'+p['ident'])

params = OrderedDict()
params["vary"] = ["alpha", "beta", "r_max"]
params["h1"] = [False, True]
params["l_f"] = [1.0,10.0]
params["delta"] = np.linspace(-60.0, 60.0, 61)
params["i"] = range(5)

file_prefix = 'bi_poo_vary'

do(task, params, file_prefix, create_notebooks=True)
