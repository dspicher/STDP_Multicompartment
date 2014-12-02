import json
from model import phi, phi_prime
import numpy as np
from IPython import embed
from operator import add
import itertools as it
import cPickle

def get_default(params):
    return json.load(open('default_{0}.json'.format(params),'r'))

def fixed_spiker(spikes):
    return lambda curr_t, dt, **kwargs: np.min(np.abs(curr_t-spikes)) < dt/2

def phi_spiker(phi_params=None):
    if phi_params is None:
        phi_params = get_default_phi()

    return lambda y, dt, **kwargs: phi(y[0])*dt <= np.random.rand()

def inst_backprop(curr_t, last_spike, **kwargs):
    return curr_t==last_spike

def dendr_spike_det(thresh=1.0, tau=4.0):
    return lambda y, curr_t, last_spike_dendr, **kwargs: y[1] > thresh and (curr_t-last_spike_dendr > tau)

def step_current(steps):
    return lambda t: steps[steps[:,0]<=t,1][-1]

def periodic_current(first,interval,width,dcs):
    def I_ext(t):
        if (t-first)%interval <= width/2:
            return dcs[0]
        else:
            return dcs[1]
    return I_ext


def construct_params(ids, values, prefix=''):
    ids = tuple(ids)

    base_str = prefix + reduce(add, ['_{0}_{{{1}}}'.format(ids[i],i) for i in range(len(ids))])

    combinations = it.product(*values)
    params = []
    for comb in combinations:
        curr = {id:val for (id,val) in zip(ids,comb)}
        curr['ident'] = base_str.format(*comb)
        params.append(curr)

    return params

def dump(res,ident):
    cPickle.dump(res, open('{0}.p'.format(ident),'wb'))

class Accumulator():
    def _get_size(self, key):
        if key=='y':
            return 3
        else:
            return 1

    def __init__(self, keys, sim, interval=1):
        self.keys = keys
        self.i = interval
        self.j = 0
        self.res = {}
        self.interval = interval
        eff_steps = np.arange(sim['start'], sim['end']+sim['dt'], sim['dt']*interval).shape[0]
        self.t = np.zeros(eff_steps)
        for key in keys:
            self.res[key] = np.zeros((eff_steps,self._get_size(key)))

    def add(self, curr_t, **vals):
        if np.abs(self.i-self.interval) < 1e-10:
            for key in self.keys:
                self.res[key][self.j,:] = np.atleast_2d(vals[key])

            self.t[self.j] = curr_t
            self.j += 1
            self.i = 0
        self.i += 1
