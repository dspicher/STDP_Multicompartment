
from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_phi_spiker, get_dendr_spike_det_dyn_ref
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
    neuron["phi"]['r_max'] = p["r_max"]

    my_s = {
        'start': 0.0,
        'end': 300.0,
        'dt': 0.05,
        'pre_spikes': np.arange(50.0,4000.0,250.0),
        'I_ext': lambda t:0.0
        }

    seed = int(int(time.time()*1e8)%1e9)
    accs = [PeriodicAccumulator(['weight'], interval=10)]
    accums = run(my_s, lambda **kwargs:False, get_dendr_spike_det_dyn_ref(p["thresh"], p["tau_ref_0"], p["theta_0"]), accs, seed=seed, neuron=neuron, voltage_clamp=True, U_clamp=p['Uclamp'])

    dump(accums,p['ident'])

params = OrderedDict()
params["alpha"] = [-34.0]
params["beta"] = [0.25]
params["r_max"] = [0.025]
params["thresh"] = [-20.0]
params["tau_ref_0"] = [5.0]
params["theta_0"] = [2.5]
for key in params.keys():
    val = params[key][0]
    params[key] = [0.95*val, val, 1.05*val]
params["Uclamp"] = np.linspace(-40.0,0.0,5)

file_prefix = 'artola_wiggle'

do(task, params, file_prefix, prompt=False, withmp=True)
