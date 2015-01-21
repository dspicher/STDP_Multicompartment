
from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_phi_spiker, get_dendr_spike_det
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
    learn['eta'] = p['eps']*p['eta_factor']
    learn['eps'] = p['eps']
    learn['tau_delta'] = 2.0

    neuron = get_default("neuron")
    neuron["phi"]["a"] = p["a"]

    my_s = {
        'start': 0.0,
        'end': 600.0,
        'dt': 0.05,
        'pre_spikes': np.array([250.0]),
        'I_ext': get_periodic_current(250.0, 1000.0, 0.8, 100.0)
        }

    accs = [PeriodicAccumulator(get_all_save_keys(), my_s,interval=1), BooleanAccumulator(['spike', 'dendr_spike', 'pre_spike'])]

    if p['d_s'] == 'bp':
        accums = run(my_s, get_phi_spiker(), get_inst_backprop(), accs, seed=int(time.time()), learn=learn, neuron=neuron)
    else:
        accums = run(my_s, get_phi_spiker(), get_dendr_spike_det(thresh=-45.0), accs, seed=int(time.time()), learn=learn, neuron=neuron)

    dump(accums,p['ident'])

params = OrderedDict()
params['d_s'] = ['bp','sd']
params['eta_factor'] = [1e-2,1e-3]
params['eps'] = [1e-3,1e-4]
params['delta'] = np.array([-30,-25,-20,-15,-10,-5,-2,-1,1,2,5,10,15,20,25,30])
params['tau_delta'] = [2.0,5.0,10.0,20.0]
params["a"] = [0.3,0.32,0.35,0.4]

file_prefix = 'single_shot_stdp'

do(task, params, file_prefix, prompt=False)
