from model import phi, phi_prime
import numpy as np
from IPython import embed
from helper import get_default

def get_all_save_keys():
    return ['g',
            'syn_pots_sum',
            'y',
            'spike',
            'dendr_pred',
            'h',
            'PIV',
            'pos_PIV',
            'neg_PIV',
            'dendr_spike',
            'pre_spike',
            'weight',
            'weight_update',
            'delta',
            'I_ext']

def get_fixed_spiker(spikes):
    return lambda curr, dt, **kwargs: spikes.shape[0] > 0 and np.min(np.abs(curr['t']-spikes)) < dt/2

def get_phi_spiker(neuron=None):
    if neuron is None:
        neuron = get_default("neuron")
    return lambda curr, dt, **kwargs: phi(curr['y'][0], neuron)*dt >= np.random.rand()

def get_inst_backprop():
    def inst_backprop(curr, last_spike, **kwargs):
        return np.isclose(curr['t'], last_spike['t'], atol=1e-10, rtol=1e-10)
    return inst_backprop

def get_dendr_spike_det(thresh, tau_ref=10.0):
    def dendr_spike_det(curr, last_spike_dendr, **kwargs):
        return curr['y'][1] > thresh and (curr['t']-last_spike_dendr['t'] > tau_ref)
    return dendr_spike_det

def get_dendr_spike_det_dyn_ref(thresh, tau_ref_0, theta_0):
    def dendr_spike_det_dyn_ref(curr, last_spike_dendr, **kwargs):
        if curr['y'][1] > thresh:
            curr_ref = tau_ref_0*np.exp(-(curr['y'][1]-thresh)/theta_0)
            return curr['t']-last_spike_dendr['t'] > curr_ref
        else:
            return False
    return dendr_spike_det_dyn_ref

def step_current(steps):
    return lambda t: steps[steps[:,0]<=t,1][-1]

def get_periodic_current(first, interval, width, dc_on, dc_off=0.0):
    def I_ext(t):
        if t >= (first-width/2) and np.mod(t-first+width/2,interval) < width:
            return dc_on
        else:
            return dc_off
    return I_ext
