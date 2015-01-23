
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

    first_spike = 100.0
    interval = 1000.0/p['freq']
    end = 10/p['freq']*1000.0+600.0
    my_s = {
        'start': 0.0,
        'end': end,
        'dt': 0.05,
        'pre_spikes': np.arange(first_spike,end,interval)+p['delta'],
        'I_ext': get_periodic_current(first_spike, interval, 0.8, 100.0)
        }

    accs = [PeriodicAccumulator(get_all_save_keys(), my_s,interval=1), BooleanAccumulator(['spike', 'dendr_spike'])]

    accums = run(my_s, get_phi_spiker(), get_inst_backprop(), accs, seed=int(time.time()), learn=learn)

    dump(accums,p['ident'])

params = OrderedDict()
params['eta_factor'] = [1e-3]
params['eps'] = [1e-3]
params['delta'] = np.array([-30,-20,-10,-5,5,10,20,30])
params['freq'] = [1,5]

file_prefix = 'new_model_inspect_stdp'

do(task, params, file_prefix)
