from simulation import run
from util import fixed_spiker
import numpy as np
from IPython import embed
from pylab import *

my_s = {
    'start': 0.0,
    'end': 100.0,
    'dt': 0.05,
    'pre_spikes': np.array([25,75])
    }

ys, weight, gs = run(my_s, fixed_spiker(np.array([50.0])))

plot(ys)
show()

embed()
"""
pres = np.arange(0,101,1)
last_w = np.zeros(pres.shape)
for idx, pre in enumerate(pres):
    print pre
    last_w[idx] = run(pre)


last_w = last_w - 1e-4
last_w = last_w/np.max(np.abs(last_w))

plt.plot(pres,last_w)
plt.xticks(np.arange(0,101,10))
plt.show()
"""
