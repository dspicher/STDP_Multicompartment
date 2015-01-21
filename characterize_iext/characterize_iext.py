
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
        'end': 30.0,
        'dt': 0.05,
        'pre_spikes': np.array([]),
        'I_ext': get_periodic_current(10.0,100.0,p['width'],p['dc'])
        }

    accs = [PeriodicAccumulator(['y'], my_s)]

    accums = run(my_s, lambda **kwargs: False, lambda **kwargs: False, accs, seed=int(time.time()))

    dump(accums,p['ident'])

params = OrderedDict()
params['width'] = np.linspace(0.2,2.0,10)
params['dc'] = np.linspace(5,100,20)

file_prefix = 'characterize_iext'

do(task, params, file_prefix, prompt=False, withmp=False)
