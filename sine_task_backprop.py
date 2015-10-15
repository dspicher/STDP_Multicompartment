
import cPickle
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

from helper import (BooleanAccumulator, PeriodicAccumulator, do, dump,
                    get_default)
from simulation import run
from util import (get_all_save_keys, get_fixed_spiker, get_inst_backprop,
                  get_periodic_current, get_phi_spiker, get_phi_U_learner)


def task((repetition_i, p)):

    n_syn = p["n_syn"]

    learn = get_default("learn")
    learn["eps"] = 1e-1 / (1.0 * n_syn)
    learn["eta"] = learn["eps"]*p["eps_factor"]

    neuron = get_default("neuron")
    neuron["phi"]["alpha"] = p["alpha"]
    neuron["phi"]["beta"] = p["beta"]
    neuron["phi"]["r_max"] = 0.1
    neuron["g_S"] = p["g_S"]

    learn_epochs = 20
    test_epochs = 20
    epochs = learn_epochs + test_epochs
    l_c = 8
    eval_c = 2
    cycles = epochs * l_c + (epochs + 1) * eval_c
    cycle_dur = 100.0
    epoch_dur = (l_c + eval_c) * cycle_dur
    t_end = cycles * cycle_dur

    exc_level = p["exc_level"]
    g_factor = 50

    def exc_soma_cond(t):
        if t % (cycle_dur * (l_c + eval_c)) < cycle_dur * eval_c or t > learn_epochs*epoch_dur:
            return 0.0
        else:
            return ((1 + np.sin(-np.pi/2 + t / t_end * cycles * 2 * np.pi)) * exc_level + exc_level) * g_factor

    def inh_soma_cond(t):
        if t % (cycle_dur * (l_c + eval_c)) < cycle_dur * eval_c or t > learn_epochs*epoch_dur:
            return 0.0
        else:
            return 4e-2 * g_factor

    dt = 0.05
    f_r = 0.01 # 10Hz
    t_pts = np.arange(0, t_end / cycles, dt)

    seed = int(int(time.time() * 1e8) % 1e9)
    poisson_spikes = [t_pts[np.random.rand(t_pts.shape[0]) < f_r * dt] for _ in range(n_syn)]
    poisson_spikes = [[] if spikes.shape[0] == 0 else np.concatenate(
        [np.arange(spike, t_end, cycle_dur) for spike in spikes]) for spikes in poisson_spikes]
    for train in poisson_spikes:
        train.sort()

    my_s = {
        'start': 0.0,
        'end': t_end,
        'dt': dt,
        'pre_spikes': poisson_spikes,
        'syn_cond_soma': {'E': exc_soma_cond, 'I': inh_soma_cond},
        'I_ext': lambda t: 0.0
    }

    phi_spiker = get_phi_spiker(neuron)

    # deprecated
    def my_spiker(curr, dt, **kwargs):
        # we want no spikes in eval cycles
        if curr['t'] % (cycle_dur * (l_c + eval_c)) < cycle_dur * eval_c:
            return False
        else:
            return phi_spiker(curr, dt, **kwargs)

    accs = [PeriodicAccumulator(get_all_save_keys(), interval=20, y_keep = 3), BooleanAccumulator(['spike'])]
    accums = run(my_s, phi_spiker, get_inst_backprop(),
                 accs, neuron=neuron, seed=seed, learn=learn)

    dump((seed, accums), 'sine_task_backprop/' + p['ident'])


params = OrderedDict()
params["n_syn"] = [50]
params["exc_level"] = [7e-3]
params["g_S"] = [0.0, 0.25, 0.5]
params["alpha"] = [-50.0]
params["beta"] = [0.35]
params["eps_factor"] = [1e-3]

file_prefix = 'sine_task_backprop_unlearn'

do(task, params, file_prefix, withmp=True, create_notebooks=True)
