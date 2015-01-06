from util import fixed_spiker, inst_backprop, periodic_current, dendr_spike_det, get_all_save_keys
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt




my_s = {
    'start': 0.0,
    'end': 100.0,
    'dt': 0.05,
    'pre_spikes': np.array([]),
    'I_ext': lambda t: 0.0
    }

post_spikes = np.array([50.0])

#my_s['I_ext'] = periodic_current(100.0, 200.0, 0.2, p['I'])

accs = [PeriodicAccumulator(get_all_save_keys(), my_s)]

acc = run(my_s, fixed_spiker(post_spikes), inst_backprop, accs)[0]

plt.plot(acc.t,acc.res['y'])
plt.ylim([-80,10])
plt.show()
#embed()
