
from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_phi_spiker, get_dendr_spike_det, get_fixed_spiker
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time

eps_stim_low = -3.0
eps_stim_high = -1.2

def task((repetition_i,p)):

    eps_stim = eps_stim_low+(eps_stim_high-eps_stim_low)*np.random.rand()

    learn = get_default("learn")
    learn["eta"] = [1e-7, 0.0]
    learn["eps"] = [p["eps_learn"],10**eps_stim]

    neuron = get_default("neuron")
    neuron["phi"]["alpha"] = -52.0
    neuron["phi"]["beta"] = 0.25
    neuron["phi"]["r_max"] = 0.35


    post_spikes = np.arange(40.0,2000.0,20.0)
    pre_spikes_learn = post_spikes - 10.0
    pre_spikes_stim = post_spikes - p["stim_delta"]

    my_s = {
        'start': 0.0,
        'end': 2000.0,
        'dt': 0.05,
        'pre_spikes': [pre_spikes_learn, pre_spikes_stim],
        'I_ext': lambda t: 0.0
        }

    prob = p["prob"]


    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(['weights'], interval=100)]
    accums = run(my_s, get_fixed_spiker(post_spikes), get_dendr_spike_det(p["thresh"]), accs, neuron=neuron, seed=seed, learn=learn, p_backprop=prob)


    dump((eps_stim,accums),p['ident'])

params = OrderedDict()
params["eps_learn"] = [1e-3, 2e-3,5e-3]
params["thresh"] = [-55.0,-56.0]
params["stim_delta"] = [15.0, 20.0]
params["prob"] = [0.35, 0.375, 0.4]
params["i"] = range(100)


file_prefix = 'switch_rescue_ltp_sample'

do(task, params, file_prefix, prompt=False, withmp=True, create_notebooks=True)
