
import cPickle
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

from helper import (BooleanAccumulator, PeriodicAccumulator, do, dump,
                    get_default)
from model import phi
from simulation import run
from util import (get_all_save_keys, get_fixed_spiker, get_inst_backprop,
                  get_periodic_current, get_phi_spiker, get_phi_U_learner)


def task((repetition_i, p)):

    n_syn = p["n_syn"]

    learn = get_default("learn")
    learn["eps"] = 1e-1 / (1.0 * n_syn)
    learn["eta"] = learn["eps"] * p["eps_factor"]

    neuron = get_default("neuron")
    neuron["phi"]["alpha"] = p["alpha"]
    neuron["phi"]["beta"] = p["beta"]
    neuron["phi"]["r_max"] = 0.1
    neuron["g_S"] = p["g_S"]

    learn_epochs = 2
    test_epochs = 1
    epochs = learn_epochs + test_epochs
    l_c = 4
    eval_c = 2
    cycles = epochs * l_c + (epochs + 1) * eval_c
    cycle_dur = p["cycle_dur"]
    epoch_dur = (l_c + eval_c) * cycle_dur
    t_end = cycles * cycle_dur

    g_factor = p["g_factor"]

    def exc_soma_cond(t):
        if t % (cycle_dur * (l_c + eval_c)) < cycle_dur * eval_c or t > learn_epochs * epoch_dur:
            return 0.0
        return p["exc_level"] * p["g_factor"]

    def inh_soma_cond(t):
        if t % (cycle_dur * (l_c + eval_c)) < cycle_dur * eval_c or t > learn_epochs * epoch_dur:
            return 0.0
        return 1e-1 * p["g_factor"]

    dt = 0.05
    f_r = 1.0 / cycle_dur
    t_pts = np.arange(0, t_end / cycles, dt)

    seed = int(int(time.time() * 1e8) % 1e9)

    reg_spikes = [np.arange(i+1,t_end+1,10) for i in range(n_syn)]

    poisson_spikes = [t_pts[np.random.rand(t_pts.shape[0]) < f_r * dt] for _ in range(n_syn)]
    poisson_spikes = [[] if spikes.shape[0] == 0 else np.concatenate(
        [np.arange(spike, t_end, cycle_dur) for spike in spikes]) for spikes in poisson_spikes]
    for train in poisson_spikes:
        train.sort()

    my_s = {
        'start': 0.0,
        'end': t_end,
        'dt': dt,
        'pre_spikes': reg_spikes,
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

    if p["wiggle"] is None:
        dendr_predictor = phi
    else:
        us = np.linspace(-100,20,1000)
        ampl = neuron["phi"]["r_max"]
        phis = phi(us, neuron)
        alphas = []
        for i in range(p["wiggle"]):
            alphas.append(us[phis > (i+0.5)*ampl/p["wiggle"]][0])
        r_m = neuron['phi']['r_max']/p["wiggle"]
        def dendr_predictor(V, neuron):
            return np.sum([phi(V,{'phi':{'alpha': al, 'beta':p["beta_wiggle"], 'r_max': r_m}}) for al in alphas])


    accs = [PeriodicAccumulator(get_all_save_keys(), interval=20,
                                y_keep=3), BooleanAccumulator(['spike'])]
    accums = run(my_s, get_fixed_spiker(np.array([])), get_phi_U_learner(neuron, dt, p["exc_decrease"]),
                 accs, neuron=neuron, seed=seed, learn=learn, dendr_predictor=dendr_predictor)

    dump((seed, accums), 'wiggle_test/' + p['ident'])


params = OrderedDict()
params["n_syn"] = [10]
params["g_S"] = [0.5]
params["alpha"] = [-60.0]
params["beta"] = [0.25]
params["eps_factor"] = [1e-2]
params["g_factor"] = [10]
params["exc_decrease"] = [0.99, 0.95]
params["exc_level"] = np.linspace(1e-2, 5e-2, 21)
params["cycle_dur"] = [100.0]
params["wiggle"] = [4, 5]
params["beta_wiggle"] = [2]

file_prefix = 'wiggle_test'

do(task, params, file_prefix, withmp=True, create_notebooks=True)
