import numpy as np
from scipy import integrate, random
from IPython import embed
from helper import get_default
from util import step_current
from model import get_spike_currents, phi, phi_prime, urb_senn_rhs

def run(sim, spiker, spiker_dendr, accumulators, neuron=None, learn=None, normalizer=None, **kwargs):

    np.random.seed(kwargs.get('seed',0))

    if neuron is None:
        neuron = get_default("neuron")

    # dendritic spike threshold
    # heuristic formular which gives one for roberts shifted system
    thresh = neuron["E_L"] + (neuron["E_E"] - neuron["E_L"]) / (4.0+2.0/3.0)

    if learn is None:
        learn = get_default("learn")

    if normalizer is None:
        normalizer = lambda x: np.max(np.array([x,0.0]))

    I_ext = sim.get('I_ext', step_current(np.array([[sim['start'],0.0]])))

    pre_spikes = sim.get('pre_spikes',np.array([]))

    t_start, t_end, dt = sim['start'], sim['end'], sim['dt']

    n_steps = int((t_end-t_start)/dt)+1

    curr = {'t':t_start,
            'y': np.array([neuron["E_L"], neuron["E_L"], neuron['g_D']/(neuron['g_D']+neuron['g_L'])*neuron["E_L"], 0.0, 0.0, 0.0])}
    last_spike = {'t': float("-inf"), 'y':curr['y']}
    last_spike_dendr = {'t': float("-inf"), 'y':curr['y']}

    weight = learn['eps']

    g_E_D = 0.0
    syn_pots_sum = 0.0
    PIV = 0.0

    vals = {'g':0.0,
            'syn_pots_sum':0.0,
            'y':curr['y'],
            'spike':0.0,
            'dendr_pred':0.0,
            'h':0.0,
            'PIV': 0.0,
            'dendr_spike':0.0,
            'weight':weight,
            'weight_update':0.0,
            'I_ext':0.0}

    for acc in accumulators:
        acc.add(curr['t'], **vals)

    while curr['t'] < t_end - dt/2:

        g_E_D = g_E_D + np.sum(np.isclose(pre_spikes, curr['t']))*weight
        g_E_D = g_E_D - dt*g_E_D/neuron['tau_s']

        syn_pots_sum = np.sum(np.exp(-(curr['t'] - pre_spikes[pre_spikes <= curr['t']])/neuron['tau_s']))

        args=(curr['t']-last_spike['t'], g_E_D, syn_pots_sum, I_ext(curr['t']), neuron, learn, PIV,)
        curr['y'] = integrate.odeint(urb_senn_rhs, curr['y'], np.array([curr['t'], curr['t']+dt]), hmax=dt, args=args)[1,:]

        curr['t'] += dt

        if curr['t'] - last_spike['t'] < neuron['tau_ref']:
            does_spike = False
        else:
            does_spike = spiker(curr=curr, dt=dt)

        if does_spike:
            last_spike = {'t': curr['t'], 'y': curr['y']}

        dendr_pred = phi(curr['y'][2], neuron)
        if neuron["phi"]["function"] == 'exp':
            h = neuron["phi"]["a"]
        else:
            h = phi_prime(curr['y'][2], neuron)/phi(curr['y'][2], neuron)

        dendr_spike = spiker_dendr(curr=curr, last_spike=last_spike, last_spike_dendr=last_spike_dendr)

        if dendr_spike:
            last_spike_dendr = {'t': curr['t'], 'y': curr['y']}

        PIV = (neuron['delta_factor']*float(dendr_spike)/dt - dendr_pred)*h*curr['y'][4]

        weight_update = learn['eta']*curr['y'][5]
        weight += weight_update

        weight = normalizer(weight)
        if weight < 0:
            embed()

        vals = {'g':g_E_D,
                'syn_pots_sum':syn_pots_sum,
                'y':curr['y'],
                'spike':float(does_spike),
                'dendr_pred':dendr_pred,
                'h':h,
                'PIV': PIV,
                'dendr_spike':float(dendr_spike),
                'weight':weight,
                'weight_update':weight_update,
                'I_ext':I_ext(curr['t'] - dt)}

        for acc in accumulators:
            acc.add(curr['t'], **vals)

    return accumulators
