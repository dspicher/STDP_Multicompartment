
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
    neuron["phi"]["r_max"] = p["r0"]
    neuron["phi"]["k"] = p["k"]
    neuron["phi"]["beta"] = p["beta"]
    neuron["thresh"] = p["thresh"]

    my_s = {
        'start': 0.0,
        'end': 2000.0,
        'dt': 0.05,
        'pre_spikes': np.array([]),
        'I_ext': lambda t: get_periodic_current(249.25, 250.0, 1.5, 80.0)(t) + get_periodic_current(250.5, 250.0, 1.0, 50.0)(t)
        }
    
    
    spiker = get_phi_spiker(neuron)
    
    accs = [PeriodicAccumulator(get_all_save_keys(), interval=1), BooleanAccumulator(['spike', 'dendr_spike', 'pre_spike'])]
        
    accums = run(my_s, get_phi_spiker(neuron), get_inst_backprop(), accs, seed=int(time.time()*1e6%1e8), neuron=neuron)
    
    print p['ident']
    print accums[1].res['spike']
        
    dump(accums,p['ident'])

params = OrderedDict()
params['r0'] = [1.0]
params['k'] = [10]
params['beta'] = [30.0]
params['thresh'] = [30]

file_prefix = 'sigm_firing'

do(task, params, file_prefix, prompt=False, withmp=True)
