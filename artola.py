
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
    neuron["phi"]['alpha'] = p["alpha"]
    neuron["phi"]['beta'] = p["beta"]

    my_s = {
        'start': 0.0,
        'end': 500.0,
        'dt': 0.05,
        'pre_spikes': np.linspace(0.0,500.0,p["n_spikes"]+2)[1:-1],
        'I_ext': lambda t:0.0
        }

    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(get_all_save_keys(), interval=1)]
    accums = run(my_s, lambda **kwargs:False, get_dendr_spike_det(thresh=p['thresh'],tau_ref=p['tau_ref']), accs, seed=seed, neuron=neuron, voltage_clamp=True, U_clamp=p['Uclamp'])

    dump(accums,p['ident'])

params = OrderedDict()
params["alpha"] = [-40]
params["beta"] = np.linspace(0.15,0.4,6)
params["n_spikes"] = [1,5,10]
params["thresh"] = [-40,-30,-20]
params["tau_ref"] = [2.0,5.0,10.0,20.0]
params["Uclamp"] = np.linspace(-70,0,8)

file_prefix = 'artola'

do(task, params, file_prefix, prompt=False, withmp=True)
