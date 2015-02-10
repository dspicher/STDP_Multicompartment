
from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_phi_spiker, get_dendr_spike_det_dyn_ref, get_fixed_spiker
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time

good_params = [  (1e-3, -50, 0.20, 0.5),
            (1e-3, -50, 0.25, 0.5),
            (1e-3, -50, 0.30, 0.5),
            (1e-3, -55, 0.3, 0.3),
            (1e-3, -55, 0.3, 0.35),
            (1e-3, -55, 0.3, 0.4),
            (1e-3, -55, 0.5, 0.2),
            (5e-3, -60, 0.6, 0.2),
            (5e-3, -60, 0.7, 0.2),
            (5e-3, -60, 0.4, 0.1)]

def task((repetition_i,p)):
    (eps, alpha, beta,r_max) = good_params[p["idx"]]

    learn = {}
    learn['eta'] = p["eta"]*eps
    learn['eps'] = eps
    learn['tau_delta'] = 2.0

    neuron = get_default("neuron")
    neuron["phi"]["function"] = "sigm"
    neuron["phi"]['r_max'] = r_max
    neuron["phi"]['alpha'] = alpha
    neuron["phi"]['beta'] = beta

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
params["idx"] = range(10)
params["eta"] = [1e-4,1e-3]
params["delta"] = [-30,-20,-10,10,20,30]

file_prefix = 'single_shot_stdp_phi_sigm_prescr_good_params'

do(task, params, file_prefix, prompt=False, withmp=True)
