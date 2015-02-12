
from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_phi_spiker, get_dendr_spike_det_dyn_ref
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time

best_params = cPickle.load(open('best_keys_wiggle.p','rb'))
count = len(best_params)

def task((repetition_i,p)):
    
    (alpha,beta,r_max,thresh,tau_ref_0,theta_0) = best_params[p["i"]]
    
    learn = get_default("learn")
    learn["eta"] = 1e-7

    neuron = get_default("sigm_neuron")
    neuron["phi"]['alpha'] = alpha
    neuron["phi"]['beta'] = beta
    neuron["phi"]['r_max'] = r_max

    my_s = {
        'start': 0.0,
        'end': 10000.0,
        'dt': 0.05,
        'pre_spikes': np.arange(50.0,10000.0,250.0),
        'I_ext': lambda t:0.0
        }

    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(['weight'], interval=100)]
    accums = run(my_s, lambda **kwargs:False, get_dendr_spike_det_dyn_ref(thresh, tau_ref_0, theta_0), accs, seed=seed, neuron=neuron, learn=learn, voltage_clamp=True, U_clamp=p['Uclamp'])

    dump(accums,p['ident'])

params = OrderedDict()
params["i"] = range(count)
params["Uclamp"] = np.linspace(-40.0,0.0,5)

file_prefix = 'artola_wiggle_best_params_long'

do(task, params, file_prefix, prompt=False, withmp=True)
