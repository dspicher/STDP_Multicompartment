"""
Here we reproduce experiments reported in
"Rate, Timing, and Cooperativity Jointly Determine
Cortical Synaptic Plasticity"
Per Jesper Sjostrom, Gina G. Turrigiano, and Sacha B. Nelson
Neuron, 2001

Specifically, we investigate the dependence of plasticity on the frequency
with which pre-post stimulation pulses are delivered at either +10ms or -10ms.
(Figure 1D and 7B, combined data also shown in Figure 8A).
The data can be found in the "experimental_data" folder.

Approximate runtime on an Intel Xeon X3470 machine (4 CPUs, 8 threads):
4min

Running this file should produce 10 .p files.

Afterwards, code in the corresponding
IPython notebook will produce a figure showing experimental data and
simulation results next to each other.
"""

from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_fixed_spiker, get_dendr_spike_det
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time


def fit((repetition_i,p)):

    vary = {"alpha":(-2.0,2.0),
              "beta":(-0.1,0.2),
              "r_max":(-0.05,0.15)}

    learn = get_default("learn")
    if p["h1"]:
        learn['eta'] *= 0.125*p["l_f"]
    else:
        learn["eta"] *= 0.8*p["l_f"]
    learn["eps"] *= p["l_f"]

    n_spikes = 5.0

    neuron = get_default("neuron")
    neuron["phi"]['r_max'] = 0.2
    neuron["phi"]['alpha'] = -54.0
    neuron["phi"]['beta'] = 0.2
    
    neuron["phi"][p["vary"]] += np.linspace(vary[p["vary"]][0], vary[p["vary"]][1], 5)[p["i"]]

    freq = p["freq"]
    delta = p["delta"]

    first_spike = 1000.0/(2*freq)
    isi = 1000.0/freq
    t_end = 1000.0*n_spikes/freq

    spikes = np.arange(first_spike, t_end, isi)
    pre_spikes = spikes + delta

    my_s = {
        'start': 0.0,
        'end': t_end,
        'dt': 0.05,
        'pre_spikes': [pre_spikes],
        'I_ext': lambda t: 0.0
        }

    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(['weights'], interval=100)]
    if p["h1"]:
        accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det(-50.0), accs, seed=seed, learn=learn, neuron=neuron, h=1.0)
    else:
        accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det(-50.0), accs, seed=seed, learn=learn, neuron=neuron)

    dump(accums,'sjostrom/'+p['ident'])

params = OrderedDict()
params["vary"] = ["alpha", "beta", "r_max"]
params["h1"] = [False, True]
params["l_f"] = [1.0,10.0]
params["freq"] = np.array([1.0,10.0,20.0,40.0,50.0])
params["delta"] = np.array([-10.0,10.0])
params["i"] = range(5)


file_prefix = 'sjostrom'

do(fit, params, file_prefix, withmp=True)


def overfit((repetition_i,p)):

    learn = {}
    learn['eta'] = 8e-7
    learn['eps'] = 1e-3
    learn['tau_delta'] = 2.0

    n_spikes = 5.0

    neuron = get_default("neuron")
    neuron["phi"]['r_max'] = p["r_max"]
    neuron["phi"]['alpha'] = p["alpha"]
    neuron["phi"]['beta'] = p["beta"]
    neuron["g_L"] = p["g_L"]

    freq = p["freq"]
    delta = p["delta"]

    first_spike = 1000.0/(2*freq)
    isi = 1000.0/freq
    t_end = 1000.0*n_spikes/freq

    spikes = np.arange(first_spike, t_end, isi)
    pre_spikes = spikes + delta

    my_s = {
        'start': 0.0,
        'end': t_end,
        'dt': 0.05,
        'pre_spikes': [pre_spikes],
        'I_ext': lambda t: 0.0
        }

    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(['weights'], interval=100)]
    accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det(-50.0), accs, seed=seed, learn=learn, neuron=neuron)

    dump(accums,'sjostrom/'+p['ident'])


params = OrderedDict()
params['alpha'] = [-54.0]
params["beta"] = [0.1]
params["g_L"] = [0.03]
params["r_max"] = [0.071]
params["freq"] = np.array([1.0,10.0,20.0,40.0,50.0])
params["delta"] = np.array([-10.0,10.0])


file_prefix = 'sjostrom_overfit'

do(overfit, params, file_prefix, withmp=True)
