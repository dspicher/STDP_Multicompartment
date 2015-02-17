
from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_fixed_spiker, get_dendr_spike_det_dyn_ref
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time


def task((repetition_i,p)):
    
    N = 20

    learn = {}
    learn['eta'] = 1e-6
    learn['eps'] = 1e-3
    learn['tau_delta'] = 2.0

    neuron = get_default("neuron")
    neuron["phi"]['alpha'] = p["alpha"]
    neuron["phi"]['beta'] = p["beta"]
    neuron["g_L"] = p["g_L"]

    t_end = 250.0
    pre_spikes = np.array([40.0])
    my_s = {
        'start': 0.0,
        'end': t_end,
        'dt': 0.05,
        'pre_spikes': pre_spikes,
        'I_ext': lambda t: 0.0
        }
    
    history = []
    
    high_r_max = 1.0
    neuron["phi"]["r_max"] = high_r_max
    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(['weight'], interval=10)]
    accums = run(my_s, get_fixed_spiker(np.array([50.0])), get_dendr_spike_det_dyn_ref(-50.0,10.0,100.0), accs, seed=seed, learn=learn, neuron=neuron)
    ratio = accums[0].res['weight'][-1]/accums[0].res['weight'][0]
    history.append((high_r_max, ratio, seed))
    if ratio > 1.0:
        print "required r_max too high for {0}".format(p["ident"])
        dump({'hist':history},p['ident'])
        return
        
    low_r_max = 0.05
    neuron["phi"]["r_max"] = low_r_max
    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(['weight'], interval=10)]
    accums = run(my_s, get_fixed_spiker(np.array([50.0])), get_dendr_spike_det_dyn_ref(-50.0,10.0,100.0), accs, seed=seed, learn=learn, neuron=neuron)
    ratio = accums[0].res['weight'][-1]/accums[0].res['weight'][0]
    history.append((low_r_max, ratio, seed))
    if ratio < 1.0:
        print "required r_max too low for {0}".format(p["ident"])
        dump({'hist':history},p['ident'])
        return
    
    for n in range(N):
        
        middle_r_max = (low_r_max+high_r_max)/2
        neuron["phi"]["r_max"] = middle_r_max
        seed = int(int(time.time()*1e8)%1e9)
        accs = [PeriodicAccumulator(['weight'], interval=10)]
        accums = run(my_s, get_fixed_spiker(np.array([50.0])), get_dendr_spike_det_dyn_ref(-50.0,10.0,100.0), accs, seed=seed, learn=learn, neuron=neuron)
        ratio = accums[0].res['weight'][-1]/accums[0].res['weight'][0]
        history.append((middle_r_max, ratio, seed))
        
        if ratio > 1.0:
            low_r_max = middle_r_max
        else:
            high_r_max = middle_r_max
            
    dump({'hist':history},p['ident'])


params = OrderedDict()
params['alpha'] = np.linspace(-60,-40,11)
params["beta"] = np.linspace(0.1,0.7,7)
params["g_L"] = [0.05, 0.03]

file_prefix = 'sj_find_no_ltp'

do(task, params, file_prefix, prompt=False, withmp=True)

