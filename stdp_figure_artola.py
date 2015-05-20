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

from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_phi_spiker, get_dendr_spike_det_dyn_ref
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time

def task((repetition_i,p)):

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

    dump(accums,p['ident'])

params = OrderedDict()
params["alpha"] = [-34.0]
params["beta"] = [0.25]
params["r_max"] = [0.015]
params["thresh"] = [-20.0]
params["tau_ref_0"] = [5.0]
params["theta_0"] = [2.5]
params["Uclamp"] = np.linspace(-40.0,0.0,9)

file_prefix = 'stdp_figure_artola'

do(task, params, file_prefix, prompt=False, withmp=False)
