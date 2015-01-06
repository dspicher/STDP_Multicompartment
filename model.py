import numpy as np
from exceptions import NotImplementedError
from IPython import embed


def get_spike_currents(U, t_post_spike, neuron):
    current = 0.0
    if 0.0 <= t_post_spike < neuron['t_rise']:
        current += neuron['g_Na']*(neuron['E_Na'] - U)
    if neuron['t_rise'] <= t_post_spike < neuron['t_fall']:
        current += neuron['g_K']*(neuron['E_K'] - U)
    return current

def phi(U, neuron):
    phi_params = neuron['phi']
    if phi_params['function'] == 'exp':
        return phi_params['r0']*np.exp(phi_params['a']*(U-phi_params['shift']))
    elif phi_params['function'] == 'sigm':
        shift = neuron['E_L']
        thresh = neuron['E_I'] - neuron['E_K']
        return phi_params['r_max']/(1+phi_params['k']*np.exp(phi_params['beta']*(1-(U-shift)/thresh)))
    else:
        raise NotImplementedError

def phi_prime(U, neuron):
    phi_params = neuron['phi']
    if phi_params['function'] == 'exp':
        return phi_params['a']
    elif phi_params['function'] == 'sigm':
        thresh = neuron['E_I'] - neuron['E_K']
        shift = neuron['E_L']
        exp_term = np.exp(phi_params['beta']*(1-(U-shift)/thresh))
        return phi_params['beta']*exp_term*phi_params['k']*phi_params['r_max']/(((1+exp_term*phi_params['k'])**2)*thresh)
    else:
        raise NotImplementedError

def urb_senn_rhs(y, t, t_post_spike, g_E_D, syn_pots_sum, I_ext, neuron, learn, PIV):
    # y=[U,V,dVdw,deltaW]
    dy = np.zeros(4)

    dy[0] = -neuron['g_L']*(y[0]-neuron['E_L']) + neuron['g_D']*(y[1]-y[0]) + I_ext
    if t_post_spike <= neuron['t_fall']:
        dy[0] = dy[0] + get_spike_currents(y[0],t_post_spike, neuron)

    dy[1] = -neuron['g_L']*(y[1]-neuron['E_L']) + neuron['g_S']*(y[0]-y[1]) + g_E_D*(neuron['E_E']-y[1])

    dy[2] = -(neuron['g_L']+(neuron['g_S']*neuron['g_L'])/(neuron['g_D']+neuron['g_L'])+g_E_D)*y[2] + (neuron['E_E']-y[1])*syn_pots_sum

    dy[3] = (PIV - y[3])/learn['tau_delta']

    return dy
