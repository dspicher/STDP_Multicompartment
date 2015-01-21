import sys
sys.path.append("C:\\Users\\dominik\\Dropbox\\scholar\\phd\\stdp_modeling\\py_stdp_match\\repo")

from util import get_phi_spiker, get_inst_backprop, get_periodic_current, get_dendr_spike_det, get_all_save_keys
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time

def test_stdp(neuron):
    for delta in [-20,-10,10,20]:

        my_s = {
            'start': 0.0,
            'end': 1000.0,
            'dt': 0.05,
            'pre_spikes': np.arange(100.0,1000.0,300.0)+delta,
            'I_ext': get_periodic_current(100.0, 300.0, 1.0, 90.0)
            }

        accs = [PeriodicAccumulator(get_all_save_keys(), my_s)]


        acc = run(my_s, get_phi_spiker(), get_inst_backprop(), accs, neuron=neuron,seed=3)[0]

        plt.figure(figsize=(12,20))
        plt.subplot(5,1,1)
        plt.plot(acc.t, acc.res['y'][:,:5])
        plt.subplot(5,1,2)
        plt.plot(acc.t, acc.res['y'][:,5])
        plt.subplot(5,1,3)
        plt.plot(acc.t,acc.res['PIV'])
        plt.subplot(5,1,4)
        plt.plot(acc.t,np.cumsum(acc.res['weight_update']))
        plt.subplot(5,1,5)
        plt.plot(acc.t,acc.res['weight'])
        plt.suptitle(str(delta))
    plt.show()

def test_spike_initiation(neuron):
    spikes = np.array([])

    for i in range(5):
        my_s = {
            'start': 0.0,
            'end': 550.0,
            'dt': 0.05,
            'pre_spikes': np.array([]),
            'I_ext': get_periodic_current(10.0, 100.0, 1.0, 90.0)
            }

        accs = [PeriodicAccumulator(get_all_save_keys(), my_s)]

        acc = run(my_s, get_phi_spiker(), get_inst_backprop(), accs, neuron=neuron,seed=int(time.time()))[0]

        print np.around(acc.t[np.squeeze(acc.res['spike'])>0.1],1)

def test_params(slope,shift):
    neuron = get_default("neuron")
    neuron['phi']['a'] = slope
    neuron['phi']['shift'] = shift
    test_spike_initiation(neuron)
    test_stdp(neuron)

test_params(3.0,-40.0)
