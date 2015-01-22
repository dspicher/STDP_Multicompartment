import numpy as np
from scipy import integrate, random
from IPython import embed
from helper import get_default
from util import step_current
from model import get_spike_currents, phi, phi_prime, urb_senn_rhs

def run(sim, spiker, spiker_dendr, accumulators, neuron=None, learn=None, normalizer=None, **kwargs):

    np.random.seed(kwargs.get('seed',0))

    if neuron is None:
        neuron = get_default('neuron')

    if learn is None:
        learn = get_default('learn')

    if normalizer is None:
        normalizer = lambda x: np.max(np.array([x,0.0]))

    I_ext = sim.get('I_ext', step_current(np.array([[sim['start'],0.0]])))

    pre_spikes = sim.get('pre_spikes',np.array([]))

    t_start, t_end, dt = sim['start'], sim['end'], sim['dt']

    curr = {'t':t_start,
            'y': np.array([neuron['E_L'], neuron['E_L'], neuron['g_D']/(neuron['g_D']+neuron['g_L'])*neuron['E_L'], 0.0, 0.0, 0.0])}
    last_spike = {'t': float('-inf'), 'y':curr['y']}
    last_spike_dendr = {'t': float('-inf'), 'y':curr['y']}

    weight = learn['eps']

    g_E_D = 0.0
    syn_pots_sum = 0.0
    PIV = 0.0
    weight_update = 0.0

    while curr['t'] < t_end - dt/2:

        # is there a presynaptic spike at curr['t']?
        curr_pre = np.sum(np.isclose(pre_spikes, curr['t'], rtol=1e-10, atol=1e-10))

        g_E_D = g_E_D + curr_pre*weight
        g_E_D = g_E_D - dt*g_E_D/neuron['tau_s']

        syn_pots_sum = np.sum(np.exp(-(curr['t'] - pre_spikes[pre_spikes <= curr['t']])/neuron['tau_s']))

        # is there a postsynaptic spike at curr['t']?
        if curr['t'] - last_spike['t'] < neuron['tau_ref']:
            does_spike = False
        else:
            does_spike = spiker(curr=curr, dt=dt)

        if does_spike:
            last_spike = {'t': curr['t'], 'y': curr['y']}

        # does the dendrite detect a spike?
        dendr_spike = spiker_dendr(curr=curr, last_spike=last_spike, last_spike_dendr=last_spike_dendr)

        if dendr_spike:
            last_spike_dendr = {'t': curr['t'], 'y': curr['y']}

        # dendritic prediction
        dendr_pred = phi(curr['y'][2], neuron)
        if neuron['phi']['function'] == 'exp':
            h = neuron['phi']['a']
        else:
            h = phi_prime(curr['y'][2], neuron)/phi(curr['y'][2], neuron)

        # update weight
        PIV = (neuron['delta_factor']*float(dendr_spike)/dt - dendr_pred)*h*curr['y'][4]
        weight_update = learn['eta']*curr['y'][5]
        weight = normalizer(weight + weight_update)

        # advance state: integrate from curr['t'] to curr['t']+dt
        curr_I = I_ext(curr['t'])
        args=(curr['t']-last_spike['t'], g_E_D, syn_pots_sum, curr_I, neuron, learn, PIV,)
        curr['y'] = integrate.odeint(urb_senn_rhs, curr['y'], np.array([curr['t'], curr['t']+dt]), hmax=dt, args=args)[1,:]
        curr['t'] += dt

        # save state
        vals = {'g':g_E_D,
                'syn_pots_sum':syn_pots_sum,
                'y':curr['y'],
                'spike':float(does_spike),
                'dendr_pred':dendr_pred,
                'h':h,
                'PIV': PIV,
                'dendr_spike':float(dendr_spike),
                'pre_spike':curr_pre,
                'weight':weight,
                'weight_update':weight_update,
                'I_ext':curr_I}
        for acc in accumulators:
            acc.add(curr['t'], **vals)

    
    for acc in accumulators:
        acc.cleanup()
        
    return accumulators
