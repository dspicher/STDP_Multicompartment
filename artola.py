"""
Here we reproduce experiments reported in
"Synaptic Activity Modulates the Induction of Bidirectional Synaptic Changes in Adult Mouse Hippocampus"
Anaclet Ngezahayo, Melitta Schachner, and Alain Artola
The Journal of Neuroscience, 2000

Specifically, we investigate plasticity when the postsynaptic neuron is voltage-clamped
at some particular voltage (Figure 2d). The data from this figure can be found in
the "experimental_data" folder.

Approximate runtime on an Intel Xeon X3470 machine (4 CPUs, 8 threads):
< 1min

Running this file should produce 9 .p files.

Afterwards, code in the corresponding
IPython notebook will produce a figure showing experimental data and
simulation results next to each other.
"""

from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_phi_spiker, get_dendr_spike_det
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time

def overfit((repetition_i,p)):

    neuron = get_default("neuron")
    neuron["phi"]['alpha'] = p["alpha"]
    neuron["phi"]['beta'] = p["beta"]
    neuron["phi"]['r_max'] = p["r_max"]

    learn = get_default("learn")
    learn["eta"] = 1e-7

    my_s = {
        'start': 0.0,
        'end': 4000.0,
        'dt': 0.05,
        'pre_spikes': [np.arange(50.0,4000.0,250.0)],
        'I_ext': lambda t:0.0
        }

    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(['weights'], interval=10)]
    accums = run(my_s, lambda **kwargs:False, get_dendr_spike_det_dyn_ref(p["thresh"], p["tau_ref_0"], p["theta_0"]), accs, seed=seed, neuron=neuron, voltage_clamp=True, U_clamp=p['Uclamp'])

    dump(accums,'artola/'+p['ident'])

params = OrderedDict()
params["alpha"] = [-34.0]
params["beta"] = [0.25]
params["r_max"] = [0.015]
params["thresh"] = [-20.0]
params["tau_ref_0"] = [5.0]
params["theta_0"] = [2.5]
params["Uclamp"] = np.linspace(-40.0,0.0,9)

file_prefix = 'artola_overfit'

#do(overfit, params, file_prefix, withmp=False, create_notebooks=False)


def fit((repetition_i,p)):

    neuron = get_default("neuron")
    neuron["phi"]['alpha'] = p["alpha"]
    neuron["phi"]['beta'] = p["beta"]
    neuron["phi"]['r_max'] = p["r_max"]

    learn = get_default("learn")
    learn["eta"] = 4e-7

    my_s = {
        'start': 0.0,
        'end': 1000.0,
        'dt': 0.05,
        'pre_spikes': [np.arange(50.0,1000.0,250.0)],
        'I_ext': lambda t:0.0
        }

    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(['weights'], interval=10)]
    if p["h1"]:
        accums = run(my_s, lambda **kwargs:False, get_dendr_spike_det(p["thresh"]), accs, seed=seed, neuron=neuron, learn=learn, voltage_clamp=True, U_clamp=p['Uclamp'], h=1.0)
    else:
        accums = run(my_s, lambda **kwargs:False, get_dendr_spike_det(p["thresh"]), accs, seed=seed, neuron=neuron, learn=learn, voltage_clamp=True, U_clamp=p['Uclamp'])

    dump(accums,'artola/'+p['ident'])

params = OrderedDict()
params["alpha"] = [-50.0, -40.0, -30.0]
params["beta"] = [0.1, 0.2]
params["r_max"] = [0.1, 0.3]
params["thresh"] = [-30.0, -40.0, -50.0]
params["Uclamp"] = np.linspace(-40.0,0.0,5)
params["h1"] = [False, True]

file_prefix = 'artola_fit'

do(fit, params, file_prefix, withmp=False, create_notebooks=True)
