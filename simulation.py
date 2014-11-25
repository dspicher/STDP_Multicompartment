import numpy as np
from scipy import integrate, random
from IPython import embed
from util import get_default
from model import get_spike_currents, phi, phi_prime, urb_senn_rhs

def run(sim, spiker, neuron=None, phi_params=None, learn=None, **kwargs):

    np.random.seed(kwargs.get('seed',0))

    if neuron is None:
        neuron = get_default("neuron")

    if phi_params is None:
        phi_params = get_default("phi")

    if learn is None:
        learn = get_default("learn")

    I_ext_steps = sim.get('I_ext', np.array([[sim['start'],0.0]]))
    def I_ext(t):
        return I_ext_steps[I_ext_steps[:,0]<=t,1][-1]

    pre_spikes = sim.get('pre_spikes',np.array([]))

    t_start, t_end, dt = sim['start'], sim['end'], sim['dt']

    n_steps = int((t_end-t_start)/dt)+1

    curr_t = t_start

    weight = np.zeros(n_steps-1)
    weight[0] = learn['eps']

    ys = np.zeros((n_steps,3))

    last_spike = float("-inf")

    g_E_D = 0.0
    gs = np.zeros(n_steps)
    gs[0] = g_E_D

    syn_pots_sum = 0.0
    curr_write_index = 1

    while curr_t < t_end - 1.5*dt:

        g_E_D = g_E_D + np.sum(np.abs(pre_spikes-curr_t)<1e-10)*weight[curr_write_index-1]
        g_E_D = g_E_D - dt*g_E_D/neuron['tau_s']

        gs[curr_write_index] = g_E_D

        syn_pots_sum = np.sum(np.exp(-(curr_t - pre_spikes[pre_spikes <= curr_t])/neuron['tau_s']))

        args=(curr_t-last_spike, g_E_D, syn_pots_sum, I_ext(curr_t), neuron,)
        ys[curr_write_index,:] = integrate.odeint(urb_senn_rhs, ys[curr_write_index-1,:], np.array([curr_t,curr_t+dt]),hmax=dt,args=args)[1,:]

        if curr_t - last_spike < neuron['tau_ref']:
            does_spike = False
        else:
            does_spike = spiker(ys[curr_write_index,:], curr_t, dt)

        if does_spike:
            last_spike = curr_t

        V_w_star = neuron['g_D']/(neuron['g_D']+neuron['g_L'])*ys[curr_write_index,1]
        dendr_pred = phi(V_w_star, phi_params)
        h = phi_prime(V_w_star, phi_params)/phi(V_w_star, phi_params)

        weight[curr_write_index] = weight[curr_write_index-1] + learn['eta']*(float(last_spike==curr_t) - dt*dendr_pred)*h*ys[curr_write_index,2]
        

        if weight[curr_write_index] < 0.0:
            weight[curr_write_index] = 0.0

        curr_t = curr_t + dt
        curr_write_index = curr_write_index + 1

    return ys, weight, gs
