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
        return np.exp(phi_params['log_pref'] + phi_params['a']*U)
    elif phi_params['function'] == 'sigm':
        return phi_params['r_max']/(1+np.exp(-phi_params['beta']*(U-phi_params['alpha'])))
    else:
        raise NotImplementedError

def phi_prime(U, neuron):
    phi_params = neuron['phi']
    if phi_params['function'] == 'exp':
        return phi_params['a']*np.exp(phi_params['log_pref'] + phi_params['a']*U)
    elif phi_params['function'] == 'sigm':
        num = np.exp((U+phi_params["alpha"])*phi_params["beta"])*phi_params["r_max"]*phi_params["beta"]
        denom = (np.exp(U*phi_params["beta"]) + np.exp(phi_params["alpha"]*phi_params["beta"]))**2
        return num/denom
    else:
        raise NotImplementedError

def urb_senn_rhs(y, t, t_post_spike, g_E_D, syn_pots_sum, I_ext, neuron):
    (U, V, V_w_star, dV_dw, dV_w_star_dw) = tuple(y)
    dy = np.zeros(5)

    # U derivative
    dy[0] = -neuron['g_L']*(U-neuron['E_L']) + neuron['g_D']*(V-U) + I_ext
    if t_post_spike <= neuron['t_fall']:
        dy[0] = dy[0] + get_spike_currents(U,t_post_spike, neuron)

    # V derivative
    dy[1] = -neuron['g_L']*(y[1]-neuron['E_L']) + neuron['g_S']*(U-V) + g_E_D*(neuron['E_E']-V)

    # V_w_star derivative
    dy[2] = -neuron['g_L']*V_w_star + neuron['g_D']*(V-V_w_star)

    # dV_dw derivative
    dy[3] = -(neuron['g_L']+neuron['g_S']+g_E_D)*dV_dw + neuron['g_S']*dV_w_star_dw + (neuron['E_E']-V)*syn_pots_sum

    # dV_w_star_dw derivative
    dy[4] = -(neuron['g_L'] + neuron['g_D'])*dV_w_star_dw + neuron['g_D']*dV_dw

    return dy
