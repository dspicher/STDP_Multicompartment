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
< 1 min

Running this file should produce 100 .p files.

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


def fit((repetition_i,p)):
    
    etas = {True: 6e-8,
            False: 30e-8}

    learn = get_default("learn")
    learn["eta"] = etas[p["h1"]]

    neuron = get_default("neuron")
    neuron["phi"]["alpha"] = -54.0
    neuron["phi"]["beta"] = 0.25
    neuron["phi"]["r_max"] = 0.2


    spikes = np.arange(20.0,301.0,20.0)

    my_s = {
        'start': 0.0,
        'end': 350.0,
        'dt': 0.05,
        'pre_spikes': [spikes-10.0],
        'I_ext': lambda t: 0.0
        }

    # 0.4 <= p <= 0.8
    prob = 0.4 + 0.4*np.random.rand()

    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(['weights'], interval=10)]
    if p["h1"]:
        accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det(-50.0), accs, neuron=neuron, seed=seed, learn=learn, p_backprop=prob, h=1.0)
    else:
        accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det(-50.0), accs, neuron=neuron, seed=seed, learn=learn, p_backprop=prob)


    dump((prob,accums),'sjostrom_switch/'+p['ident'])

params = OrderedDict()
params["h1"] = [False, True]
params["i"] = range(100)

file_prefix = 'sjostrom_switch_fit'

do(fit, params, file_prefix, withmp=True, create_notebooks=False)

def vary((repetition_i,p)):
    
    etas = {True: 6e-8,
            False: 30e-8}
            
    varies = {"alpha": np.linspace(-52.0,-56.0,3),
              "beta": np.linspace(0.15,0.25,3),
              "r_max": np.linspace(0.1,0.3,3)}

    learn = get_default("learn")
    learn["eta"] = etas[p["h1"]]

    neuron = get_default("neuron")
    neuron["phi"]["alpha"] = -54.0
    neuron["phi"]["beta"] = 0.25
    neuron["phi"]["r_max"] = 0.2
    neuron["phi"][p["vary"]] = varies[p["vary"]][p["ivary"]]


    spikes = np.arange(20.0,301.0,20.0)

    my_s = {
        'start': 0.0,
        'end': 350.0,
        'dt': 0.05,
        'pre_spikes': [spikes-10.0],
        'I_ext': lambda t: 0.0
        }

    # 0.2 <= p <= 1.0
    prob = 0.2 + 0.8*np.random.rand()

    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(['weights'], interval=10)]
    if p["h1"]:
        accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det(-50.0), accs, neuron=neuron, seed=seed, learn=learn, p_backprop=prob, h=1.0)
    else:
        accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det(-50.0), accs, neuron=neuron, seed=seed, learn=learn, p_backprop=prob)


    dump((prob,accums),'sjostrom_switch/'+p['ident'])

params = OrderedDict()
params["vary"] = ["alpha", "beta", "r_max"]
params["ivary"] = range(3)
params["h1"] = [False, True]
params["i"] = range(50)

file_prefix = 'sjostrom_switch_vary'

do(vary, params, file_prefix, withmp=True, create_notebooks=False)
