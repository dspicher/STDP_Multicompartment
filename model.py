import numpy as np

def get_spike_currents(U, t_post_spike, neuron):
    """
    this function implements our simplified action potential currents (Eq. 3)
    if t_post_spike is the time since the last spike
        0 <= t_post_spike < t_rise: Na+ conductance
        t_rise <= t_post_spike < t_fall: K+ conductance
    parameters:
    U -- the current somatic potential
    t_post_spike -- time since last spike
    neuron -- the dictionary containing neuron parameters, e.g. contents of default_neuron.json
    returns:
    action potential current
    """
    current = 0.0
    if 0.0 <= t_post_spike < neuron['t_rise']:
        current += -neuron['g_Na']*(U - neuron['E_Na'])
    if neuron['t_rise'] <= t_post_spike < neuron['t_fall']:
        current += -neuron['g_K']*(U - neuron['E_K'])
    return current

def phi(U, neuron):
    """
    the transfer function somatic voltage -> firing rate which determines the
    firing intensity of the inhomogenuous poisson process, in our case a sigmoidal function (Eq. 2)
    parameters:
    U -- the current somatic potential
    neuron -- the dictionary containing neuron parameters, e.g. contents of default_neuron.json
    returns:
    firing rate
    """
    phi_params = neuron['phi']
    return phi_params['r_max']/(1+np.exp(-phi_params['beta']*(U-phi_params['alpha'])))

def phi_prime(U, neuron):
    """
    computes the first derivative of the sigmoidal firing rate function
    """
    phi_params = neuron['phi']
    num = np.exp((U+phi_params["alpha"])*phi_params["beta"])*phi_params["r_max"]*phi_params["beta"]
    denom = (np.exp(U*phi_params["beta"]) + np.exp(phi_params["alpha"]*phi_params["beta"]))**2
    return num/denom

def urb_senn_rhs(y, t, t_post_spike, g_E_Ds, syn_pots_sums, I_ext, neuron, voltage_clamp, p_backprop):
    """
    computes the right hand side describing how the system of differential equations
    evolves in time, used for Euler integration
    parameters
    y -- the current state (see first line of code)
    t -- the current time
    t_post_spike -- time since last spike
    g_E_D -- current excitatory conductance from dendritic synapses
    syn_pots_sum -- current value of input spike train convolved with exponential decay
        see Eq. 5 and text thereafter
    I_ext -- current externally applied current (to the soma)
    neuron -- the dictionary containing neuron parameters
    voltage_clamp -- a boolean indicating whether we clamp the somatic voltage or let it evolve
    p_backprop -- a probability p where we set the conductance soma -> dendrite to zero
        with probability (1-p)
    """
    (U, V, V_w_star) = tuple(y[:3])
    dy = np.zeros(y.shape)

    # U derivative
    if voltage_clamp:
        dy[0] = 0.0
    else:
        dy[0] = -neuron['g_L']*(U-neuron['E_L']) + neuron['g_D']*(V-U) + I_ext
        if t_post_spike <= neuron['t_fall']:
            dy[0] = dy[0] + get_spike_currents(U,t_post_spike, neuron)

    # V derivative
    dy[1] = -neuron['g_L']*(V-neuron['E_L']) - np.sum(g_E_Ds)*(V-neuron['E_E'])
    if np.random.rand() <= p_backprop:
        dy[1] += -neuron['g_S']*(V-U)

    # V_w_star derivative
    dy[2] = -neuron['g_L']*(V_w_star-neuron['E_L']) + neuron['g_D']*(V-V_w_star)

    for i in range((y.shape[0]-3)/2):
        dV_dw, dV_w_star_dw = y[3+2*i], y[3+2*i+1]
        dy[3+2*i] = -(neuron['g_L']+neuron['g_S']+g_E_Ds[i])*dV_dw + neuron['g_S']*dV_w_star_dw + (neuron['E_E']-V)*syn_pots_sums[i]
        dy[3+2*i+1] = -(neuron['g_L'] + neuron['g_D'])*dV_w_star_dw + neuron['g_D']*dV_dw

    return dy
