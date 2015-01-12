
from util import get_freq_spiker, get_all_save_keys, get_periodic_current, get_inst_backprop, get_phi_spiker, get_fixed_spiker
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt

for delta in [-20,-10,10,20]:

    my_s = {
        'start': 0.0,
        'end': 1000.0,
        'dt': 0.05,
        'pre_spikes': np.arange(100.0,1000.0,300.0)+delta,
        'I_ext': get_periodic_current(100.0, 300.0, 0.8, 100.0)
        }

    accs = [PeriodicAccumulator(get_all_save_keys(), my_s)]


    acc = run(my_s, get_phi_spiker(), get_inst_backprop(), accs, seed=3)[0]

    plt.figure(figsize=(12,20))
    plt.subplot(4,1,1)
    plt.plot(acc.t, acc.res['y'][:,:5])
    plt.subplot(4,1,2)
    plt.plot(acc.t, acc.res['y'][:,5])
    plt.subplot(4,1,3)
    plt.plot(acc.t,acc.res['PIV'])
    plt.subplot(4,1,4)
    plt.plot(acc.t,np.cumsum(acc.res['weight_update']))
    plt.suptitle(str(delta))
plt.show()
