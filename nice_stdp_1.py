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
    learn['eta'] = 2.6e-6 # 40 spikes at 1e-7
    learn['eps'] = 1e-3
    learn['tau_delta'] = 2.0

    neuron = get_default("neuron")
    neuron["phi"]['r_max'] = p["r_max"]
    neuron["phi"]['alpha'] = p["alpha"]
    neuron["phi"]['beta'] = p["beta"]

    my_s = {
        'start': 0.0,
        'end': 300.0,
        'dt': 0.05,
        'pre_spikes': np.array([50.0+p["delta"]]),
        'I_ext': lambda t: 0.0
        }

    spikes = np.array([50.0])

    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(get_all_save_keys(), interval=10), BooleanAccumulator(['spike', 'dendr_spike', 'pre_spike'])]
    accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det_dyn_ref(-50.0,10.0,100.0), accs, seed=seed, learn=learn, neuron=neuron)


    dump(accums,p['ident'])

params = OrderedDict()
params["alpha"] = [-59]
params["beta"] = [0.5]
params["r_max"] = [ 0.15    ]
params["delta"] = np.linspace(-80,80,81)

file_prefix = 'nice_stdp_1'

do(task, params, file_prefix, prompt=False, withmp=False)
