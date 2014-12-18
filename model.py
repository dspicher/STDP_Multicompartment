import numpy as np


def get_spike_currents(U, t_post_spike, neuron):
    current = 0.0
    if 0.0 <= t_post_spike < neuron['t_rise']:
        current += neuron['g_Na']*(neuron['E_Na'] - U)
    if neuron['t_rise'] <= t_post_spike < neuron['t_fall']:
        current += neuron['g_K']*(neuron['E_K'] - U)
    return current

def phi(U, phi_params):
    return phi_params['r_max']/(1+phi_params['k']*np.exp(phi_params['beta']*(1-(U-phi_params["U_shift"])/phi_params['thresh'])))

def phi_prime(U, phi_params):
    exp_term = np.exp(phi_params['beta']*(1-(U-phi_params["U_shift"])/phi_params['thresh']))
    return phi_params['beta']*exp_term*phi_params['k']*phi_params['r_max']/(((1+exp_term*phi_params['k'])**2)*phi_params['thresh'])

def urb_senn_rhs(y, t, t_post_spike, g_E_D, syn_pots_sum, I_ext, neuron):
    # y=[U,V,dVdw]
    dy = np.zeros(3)

    dy[0] = -neuron['g_L']*(y[0]-neuron['E_L']) + neuron['g_D']*(y[1]-y[0]) + I_ext
    if t_post_spike <= neuron['t_fall']:
        dy[0] = dy[0] + get_spike_currents(y[0],t_post_spike, neuron)

    dy[1] = -neuron['g_L']*(y[1]-neuron['E_L']) + neuron['g_S']*(y[0]-y[1]) + g_E_D*(neuron['E_E']-y[1])

    dy[2] = -(neuron['g_L']+(neuron['g_S']*neuron['g_L'])/(neuron['g_D']+neuron['g_L'])+g_E_D)*y[2] + (neuron['E_E']-y[1])*syn_pots_sum

    return dy
