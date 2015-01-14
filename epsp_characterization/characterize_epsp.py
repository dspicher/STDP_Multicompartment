
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

    my_s = {
        'start': 0.0,
        'end': 100.0,
        'dt': 0.05,
        'pre_spikes': np.array([30]),
        'I_ext': lambda t: 0.0
        }

    learn = get_default("learn")
    learn['eps'] = p['weight']

    accs = [PeriodicAccumulator(get_all_save_keys(), my_s)]

    normalizer = lambda weight: p['weight']

    accums = run(my_s, lambda **kwargs: False, lambda **kwargs: False, accs, learn=learn, normalizer=normalizer, seed=int(time.time()))

    dump(accums,p['ident'])

params = OrderedDict()
params['weight'] = [1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1,5e-1]

file_prefix = 'characterize_epsp'

do(task, params, file_prefix, prompt=False, withmp=False)
