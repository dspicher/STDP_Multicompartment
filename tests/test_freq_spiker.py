import sys
sys.path.append("C:\\Users\\dominik\\Dropbox\\scholar\\phd\\stdp_modeling\\py_stdp_match\\repo")

from util import get_freq_spiker, get_all_save_keys, step_current
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt


my_s = {
    'start': 0.0,
    'end': 500.0,
    'dt': 0.05,
    'pre_spikes': np.array([]),
    'I_ext': step_current(np.array([[0.0,0.0],[50.0,5.0],[100.0,10.0],[150.0,15.0],[200.0,20.0]]))
    }

accs = [PeriodicAccumulator(get_all_save_keys(), my_s)]

dendr_spiker = get_freq_spiker(lambda V: -V, -60)

acc = run(my_s, lambda **kwargs:False, dendr_spiker, accs, seed=np.random.randint(100))[0]

plt.figure(figsize=(12,10))
plt.subplot(2,1,1)
plt.plot(acc.t, acc.res['y'][:,:2])
plt.subplot(2,1,2)
plt.plot(acc.t,acc.res['dendr_spike'])
plt.show()

embed()
