
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

from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_phi_spiker, get_dendr_spike_det_dyn_ref, get_fixed_spiker
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time

def task((repetition_i,p)):

    learn = {}
    learn['eta'] = 2.6e-6
    learn['eps'] = 1e-3
    learn['tau_delta'] = 2.0

    neuron = get_default("neuron")
    neuron["phi"]['r_max'] = p["r_max"]
    neuron["phi"]['alpha'] = p["alpha"]
    neuron["phi"]['beta'] = p["beta"]

    my_s = {
        'start': 0.0,
        'end': 500.0,
        'dt': 0.05,
        'pre_spikes': np.array([200.0+p["delta"]]),
        'I_ext': lambda t: 0.0
        }

    spikes = np.array([200.0])

    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(get_all_save_keys(), interval=10), BooleanAccumulator(['spike', 'dendr_spike', 'pre_spike'])]
    accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det_dyn_ref(-50.0,10.0,100.0), accs, seed=seed, learn=learn, neuron=neuron)


    dump(accums,p['ident'])

params = OrderedDict()
params["alpha"] = [-59]
params["beta"] = [0.5]
params["r_max"] = [0.15]
params["delta"] = np.linspace(-100,100,101)

file_prefix = 'stdp_figure_bi_poo'

do(task, params, file_prefix, prompt=False)
