import sys
sys.path.append("C:\\Users\\dominik\\Dropbox\\scholar\\phd\\stdp_modeling\\py_stdp_match\\repo")

from util import get_phi_spiker, get_inst_backprop, get_periodic_current, get_dendr_spike_det, get_all_save_keys
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time

def test():
    spikes = np.array([])

    for i in range(10):
        my_s = {
            'start': 0.0,
            'end': 550.0,
            'dt': 0.05,
            'pre_spikes': np.array([]),
            'I_ext': get_periodic_current(10.0, 100.0, 0.8, 90.0)
            }

        accs = [PeriodicAccumulator(get_all_save_keys(), my_s)]

        acc = run(my_s, get_phi_spiker(), get_inst_backprop(), accs, seed=int(time.time()))[0]

        print acc.t[np.squeeze(acc.res['spike'])>0.1]

test()
