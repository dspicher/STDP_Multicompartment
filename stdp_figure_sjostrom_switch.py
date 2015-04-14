"""
Here we reproduce experiments reported in
"A Cooperative Switch Determines the
Sign of Synaptic Plasticity in Distal
Dendrites of Neocortical Pyramidal Neurons"
Per Jesper Sjostrom 1,2, and Michael Hausser
Neuron, 2006

Specifically, we investigate the effects of detrimental backpropagation
on plasticity in spike-timing dependent stimulation protocols.
(Figure 3). The data from this figure can be found in the "experimental_data" folder.

Approximate runtime on an Intel Xeon X3470 machine (4 CPUs, 8 threads):
4m

Running this file should produce 100 .p files.

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

    learn = get_default("learn")
    learn["eta"] = 1e-7

    neuron = get_default("neuron")
    neuron["phi"]["alpha"] = -58.0
    neuron["phi"]["beta"] = 0.25
    neuron["phi"]["r_max"] = p["r_max"]


    spikes = np.arange(20.0,1600.0,100.0)

    my_s = {
        'start': 0.0,
        'end': 1550.0,
        'dt': 0.05,
        'pre_spikes': spikes-10.0,
        'I_ext': lambda t: 0.0
        }

    # 0.4 <= p <= 0.8
    prob = 0.4*np.random.rand()+0.4

    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(get_all_save_keys(), interval=10)]
    accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det_dyn_ref(-55.0,20.0,1.0), accs, neuron=neuron, seed=seed, learn=learn, p_backprop=prob)


    dump((prob,accums),p['ident'])

params = OrderedDict()
params["i"] = range(100)
params["r_max"] = [0.35]

file_prefix = 'stdp_figure_sjostrom_switch'

do(task, params, file_prefix, prompt=False, withmp=True)
