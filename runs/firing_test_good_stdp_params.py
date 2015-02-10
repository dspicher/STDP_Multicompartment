
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
    
    neuron = get_default("sigm_neuron")
    neuron["phi"]["r_max"] = p["r_max"]

    my_s = {
        'start': 0.0,
        'end': 20000.0,
        'dt': 0.05,
        'pre_spikes': np.array([]),
        'I_ext': get_periodic_current(50.0, 500.0, 0.8, 100.0)
        }
        
    seed = int(int(time.time()*1e8)%1e9)
    accs = [BooleanAccumulator(['spike'])]
    accums = run(my_s, get_phi_spiker(), get_inst_backprop(), accs, seed=seed, neuron=neuron)
    spks = accums[0].res['spike']
    print spks

    dump((accums, seed),p['ident'])

params = OrderedDict()
params["r_max"] = [1.25,1.5,1.75]
params['dummy'] = range(8)

file_prefix = 'firing_test_good_stdp_params'

do(task, params, file_prefix, prompt=False, withmp=True)
