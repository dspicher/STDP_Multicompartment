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

import cPickle
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

from helper import (BooleanAccumulator, PeriodicAccumulator, do, dump,
                    get_default)
from simulation import run
from util import (get_all_save_keys, get_dendr_spike_det, get_fixed_spiker,
                  get_inst_backprop, get_periodic_current)


def fit((repetition_i, p)):

    learn = get_default("learn")
    if p["h1"]:
        learn['eta'] *= 0.125 / 8.0
    else:
        learn["eta"] *= 0.1

    neuron = get_default("neuron")
    neuron["phi"]['r_max'] = 0.2
    neuron["phi"]['alpha'] = -54.0
    neuron["phi"]['beta'] = 0.25

    p_backprop = 0.75

    freq = p["freq"]
    delta = p["delta"]

    n_spikes_in_burst = 10
    burst_pause = 200.0
    bursts = 50 / n_spikes_in_burst
    burst_dur = 1000.0 * n_spikes_in_burst / freq

    first_spike = 1000.0 / (2 * freq)
    isi = 1000.0 / freq
    t_end = bursts * (burst_dur + burst_pause)

    spikes_in_burst = np.arange(first_spike, burst_dur, isi)
    spikes = np.array([])
    for i in range(bursts):
        spikes = np.concatenate((spikes, spikes_in_burst + i * (burst_dur + burst_pause)))

    pre_spikes = spikes + delta

    my_s = {
        'start': 0.0,
        'end': t_end,
        'dt': 0.05,
        'pre_spikes': [pre_spikes],
        'I_ext': lambda t: 0.0
    }

    seed = int(int(time.time() * 1e8) % 1e9)
    accs = [PeriodicAccumulator(['weights'], interval=100)]
    if p["h1"]:
        accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det(-50.0), accs,
                     seed=seed, learn=learn, neuron=neuron, h=1.0, p_backprop=p_backprop)
    else:
        accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det(-50.0), accs,
                     seed=seed, learn=learn, neuron=neuron, p_backprop=p_backprop)

    dump(accums, 'sjostrom/' + p['ident'])

params = OrderedDict()
params["h1"] = [False, True]
params["freq"] = np.array([1.0, 5.0, 10.0, 20.0, 40.0, 50.0])
params["delta"] = np.array([-10.0, 10.0])
params["j"] = range(50)


file_prefix = 'sjostrom_fit'

do(fit, params, file_prefix, withmp=True)

def vary((repetition_i, p)):

    vary = {"alpha": (-2.0, 2.0),
            "beta": (-0.1, 0.1),
            "r_max": (-0.1, 0.1),
            "p_backprop": (-5, 5)}

    learn = get_default("learn")
    if p["h1"]:
        learn['eta'] *= 0.125 / 8.0
    else:
        learn["eta"] *= 0.1

    neuron = get_default("neuron")
    neuron["phi"]['r_max'] = 0.2
    neuron["phi"]['alpha'] = -54.0
    neuron["phi"]['beta'] = 0.25

    p_backprop = 0.75

    if p["vary"] in ['alpha', 'beta', 'r_max']:
        neuron["phi"][p["vary"]] += np.linspace(vary[p["vary"]][0], vary[p["vary"]][1], 5)[p["i"]]
    else:
        p_backprop += np.linspace(vary[p["vary"]][0], vary[p["vary"]][1], 5)[p["i"]]

    freq = p["freq"]
    delta = p["delta"]

    n_spikes_in_burst = 10
    burst_pause = 200.0
    bursts = 50 / n_spikes_in_burst
    burst_dur = 1000.0 * n_spikes_in_burst / freq

    first_spike = 1000.0 / (2 * freq)
    isi = 1000.0 / freq
    t_end = bursts * (burst_dur + burst_pause)

    spikes_in_burst = np.arange(first_spike, burst_dur, isi)
    spikes = np.array([])
    for i in range(bursts):
        spikes = np.concatenate((spikes, spikes_in_burst + i * (burst_dur + burst_pause)))

    pre_spikes = spikes + delta

    my_s = {
        'start': 0.0,
        'end': t_end,
        'dt': 0.05,
        'pre_spikes': [pre_spikes],
        'I_ext': lambda t: 0.0
    }

    seed = int(int(time.time() * 1e8) % 1e9)
    accs = [PeriodicAccumulator(['weights'], interval=100)]
    if p["h1"]:
        accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det(-50.0), accs,
                     seed=seed, learn=learn, neuron=neuron, h=1.0, p_backprop=p_backprop)
    else:
        accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det(-50.0), accs,
                     seed=seed, learn=learn, neuron=neuron, p_backprop=p_backprop)

    dump(accums, 'sjostrom/' + p['ident'])

params = OrderedDict()
params["vary"] = ["alpha", "beta", "r_max", "p_backprop"]
params["h1"] = [False, True]
params["freq"] = np.array([1.0, 5.0, 10.0, 20.0, 40.0, 50.0])
params["delta"] = np.array([-10.0, 10.0])
params["i"] = range(5)
params["j"] = range(20)


file_prefix = 'sjostrom_vary'

do(vary, params, file_prefix, withmp=True)
