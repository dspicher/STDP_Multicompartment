import sys
sys.path.append("C:\\Users\\dominik\\Dropbox\\scholar\\phd\\stdp_modeling\\py_stdp_match\\repo")

from util import phi_spiker, get_inst_backprop, periodic_current, get_dendr_spike_det, get_all_save_keys
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt

def test():
    spikes = np.array([])

    for i in range(20):
        print i
        my_s = {
            'start': 30.0,
            'end': 80.0,
            'dt': 0.05,
            'pre_spikes': np.array([]),
            'I_ext': periodic_current(50.0, 100.0, 0.6, 50.0)
            }

        accs = [PeriodicAccumulator(get_all_save_keys(), my_s)]

        acc = run(my_s, phi_spiker(), get_inst_backprop(), accs, seed=np.random.randint(100))[0]

        spikes = np.concatenate((spikes, acc.t[np.squeeze(acc.res['spike'])>0.1]))

        print spikes

    assert np.isclose(np.min(spikes),np.max(spikes))
