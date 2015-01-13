
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

    end = 200.0
    my_s = {
        'start': 0.0,
        'end': end,
        'dt': 0.05,
        'pre_spikes': np.array([]),
        'I_ext': get_periodic_current(100.0, 200.0, 2.0, p['dc'])
        }

    accs = [PeriodicAccumulator(['y'], my_s)]

    accums = run(my_s, get_phi_spiker(), get_inst_backprop(), accs, seed=int(time.time()))

    dump(accums,p['ident'])

params = OrderedDict()
params['dc'] = np.linspace(40,100,21)
params['rep'] = range(10)

file_prefix = 'numeric_instability_currents'

do(task, params, file_prefix, prompt=False)
