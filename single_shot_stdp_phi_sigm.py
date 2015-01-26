
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

    res = {}

    for delta in np.array([-30,-20,-10,-5,5,10,20,30]):

        learn = {}
        learn['eta'] = 1e-6
        learn['eps'] = 1e-3
        learn['tau_delta'] = 2.0

        neuron = get_default("neuron")
        neuron["phi"]["function"] = "sigm"
        neuron["phi"]['r_max'] = 1.5
        neuron["phi"]['alpha'] = p["alpha"]
        neuron["phi"]['beta'] = p["beta"]

        my_s = {
            'start': 0.0,
            'end': 400.0,
            'dt': 0.05,
            'pre_spikes': np.array([50.0+delta]),
            'I_ext': get_periodic_current(50.0, 1000.0, 0.8, 100.0)
            }
        
        trials = []
        while True:
            seed = int(int(time.time()*1e8)%1e9)
            accs = [PeriodicAccumulator(get_all_save_keys(), interval=1), BooleanAccumulator(['spike', 'dendr_spike', 'pre_spike'])]
            accums = run(my_s, get_phi_spiker(), get_inst_backprop(), accs, seed=seed, learn=learn, neuron=neuron)
            spks = accums[1].res['spike']
            trials.append((spks,seed))
            if spks.shape[0] == 1 and abs(spks[0]-50.0) < 1.0:
                break
            if len(trials)>10:
                break
                
        res[delta] = (accums, trials)

    dump(res,p['ident'])

params = OrderedDict()
params["alpha"] = np.linspace(-60,-40,11)
params["beta"] = np.linspace(0.25,3.0,12)

file_prefix = 'single_shot_stdp_phi_sigm'

do(task, params, file_prefix, prompt=False, withmp=True)
