from model import phi, phi_prime
import numpy as np
from IPython import embed


def get_all_save_keys():
    return ['g',
            'syn_pots_sum',
            'y',
            'spike',
            'V_w_star',
            'dendr_pred',
            'h',
            'dendr_spike',
            'weight',
            'weight_update',
            'I_ext']

def fixed_spiker(spikes):
    return lambda curr_t, dt, **kwargs: spikes.shape[0] > 0 and np.min(np.abs(curr_t-spikes)) < dt/2

def phi_spiker(phi_params=None):
    if phi_params is None:
        phi_params = get_default_phi()

    return lambda y, dt, **kwargs: phi(y[0])*dt <= np.random.rand()

def inst_backprop(curr_t, last_spike, **kwargs):
    return curr_t==last_spike

def dendr_spike_det(y, curr_t, last_spike_dendr, thresh=1.0, tau=4.0, **kwargs):
    return y[1] > thresh and (curr_t-last_spike_dendr > tau)

def step_current(steps):
    return lambda t: steps[steps[:,0]<=t,1][-1]

def periodic_current(first, interval, width, dc_on, dc_off=0.0):
    def I_ext(t):
        if np.abs(t%interval - first) <= width/2 or np.isclose(first - t%interval, width/2):
            if np.isclose(t%interval,first+width/2):
                return dc_off
            else:
                return dc_on
        else:
            return dc_off
    return I_ext
