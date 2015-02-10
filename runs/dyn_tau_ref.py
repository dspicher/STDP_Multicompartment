
from util import get_all_save_keys, get_periodic_current, get_dendr_spike_det_dyn_ref, get_phi_spiker, get_dendr_spike_det
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time


def task((repetition_i,p)):

    res = {}


    neuron = get_default("sigm_neuron")

    my_s = {
        'start': 0.0,
        'end': 1000.0,
        'dt': 0.05,
        'pre_spikes': np.array([]),
        'I_ext': get_periodic_current(50.0, 200.0, 0.8, 100.0)
        }
    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(get_all_save_keys(), interval=10), BooleanAccumulator(['spike', 'dendr_spike', 'pre_spike'])]
    accums = run(my_s, get_phi_spiker(), get_dendr_spike_det_dyn_ref(p["thresh"],p["tau_ref_0"],p["theta_0"]), accs, seed=seed, neuron=neuron)

    dump(accums,p['ident'])

params = OrderedDict()
params["thresh"]=[-50,-45,-40]
params["tau_ref_0"]=[2,5,10,20]
params["theta_0"]=[1,2,5,10,20]

file_prefix = 'dyn_tau_ref_test'

do(task, params, file_prefix, prompt=False, withmp=True)
