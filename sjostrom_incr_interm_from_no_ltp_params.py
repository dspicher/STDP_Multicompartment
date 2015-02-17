
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

    learn = {}
    learn['eta'] = 1e-6
    learn['eps'] = 1e-3
    learn['tau_delta'] = 2.0
    
    ident = "sj_find_no_ltp_alpha_{0}_beta_{1}_g_L_{2}".format(p["alpha"], p["beta"], p["g_L"])
    no_ltp_hist = cPickle.load(open(ident+".p",'rb'))["hist"]
    if len(no_ltp_hist) <= 2:
        print "had no result for {0}".format(p["ident"])
        dump([],p['ident'])
        return
    
    r_max = no_ltp_hist[-1][0]

    neuron = get_default("neuron")
    neuron["phi"]['r_max'] = r_max
    neuron["phi"]['alpha'] = p["alpha"]
    neuron["phi"]['beta'] = p["beta"]
    neuron["g_L"] = p["g_L"]

    t_end = 2000.0
    
    spikes = np.arange(25.0,2000.0,50.0)
    pre_spikes = np.arange(25.0,2000.0,50.0) + p["delta"]
    my_s = {
        'start': 0.0,
        'end': t_end,
        'dt': 0.05,
        'pre_spikes': pre_spikes,
        'I_ext': lambda t: 0.0
        }
        
    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(get_all_save_keys(), interval=10), BooleanAccumulator(['spike', 'dendr_spike', 'pre_spike'])]
    accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det_dyn_ref(-50.0,10.0,100.0), accs, seed=seed, learn=learn, neuron=neuron)

    dump((accums,r_max),p['ident'])

params = OrderedDict()
params['alpha'] = np.linspace(-60,-40,11)
params["beta"] = np.linspace(0.1,0.7,7)
params["g_L"] = [0.05, 0.03]
params["delta"] = [-10.0,10.0]

file_prefix = 'sj_20_no_ltp'

do(task, params, file_prefix, prompt=False, withmp=True)

