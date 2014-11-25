import json
from model import phi, phi_prime
import numpy as np

def get_default(params):
    return json.load(open('default_{0}.json'.format(params),'r'))

def fixed_spiker(spikes):
    return lambda ys, curr_t, dt: np.min(np.abs(curr_t-spikes)) < dt/2

def phi_spiker(phi_params=None):
    if phi_params is None:
        phi_params = get_default_phi()

    return lambda ys, curr_t, dt: phi(ys[0])*dt <= np.random.rand()
