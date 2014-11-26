import numpy as np
from scipy import integrate, random
from IPython import embed
from util import get_default
from model import get_spike_currents, phi, phi_prime, urb_senn_rhs

def run(sim, spiker, spiker_dendr, accumulator, neuron=None, phi_params=None, learn=None, **kwargs):

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
    weight = learn['eps']

    last_spike = float("-inf")
    last_spike_dendr = float("-inf")

    g_E_D = 0.0
    syn_pots_sum = 0.0

    y = np.zeros(3)

    vals = {'g':0.0,
            'syn_pots_sum':0.0,
            'y':y,
            'spike':0.0,
            'V_w_star':0.0,
            'dendr_pred':0.0,
            'h':0.0,
            'dendr_spike':0.0,
            'weight':weight}

    accumulator.add(curr_t, **vals)

    while curr_t < t_end - dt:

        g_E_D = g_E_D + np.sum(np.abs(pre_spikes-curr_t)<1e-10)*weight
        g_E_D = g_E_D - dt*g_E_D/neuron['tau_s']

        syn_pots_sum = np.sum(np.exp(-(curr_t - pre_spikes[pre_spikes <= curr_t])/neuron['tau_s']))

        args=(curr_t-last_spike, g_E_D, syn_pots_sum, I_ext(curr_t), neuron,)
        y = integrate.odeint(urb_senn_rhs, y, np.array([curr_t,curr_t+dt]),hmax=dt,args=args)[1,:]

        if curr_t - last_spike < neuron['tau_ref']:
            does_spike = False
        else:
            does_spike = spiker(y=y, curr_t=curr_t, dt=dt)

        if does_spike:
            last_spike = curr_t

        V_w_star = neuron['g_D']/(neuron['g_D']+neuron['g_L'])*y[1]
        dendr_pred = phi(V_w_star, phi_params)
        h = phi_prime(V_w_star, phi_params)/phi(V_w_star, phi_params)

        dendr_spike = spiker_dendr(y=y, curr_t=curr_t, last_spike=last_spike, last_spike_dendr=last_spike_dendr)

        if dendr_spike:
            last_spike_dendr = curr_t

        weight += learn['eta']*(float(dendr_spike) - dt*dendr_pred)*h*y[2]

        if weight < 0.0:
            weight = 0.0

        curr_t = curr_t + dt

        vals = {'g':g_E_D,
                'syn_pots_sum':syn_pots_sum,
                'y':y,
                'spike':float(does_spike),
                'V_w_star':V_w_star,
                'dendr_pred':dendr_pred,
                'h':h,
                'dendr_spike':float(dendr_spike),
                'weight':weight}

        accumulator.add(curr_t, **vals)

    return accumulator
