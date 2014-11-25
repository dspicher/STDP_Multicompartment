import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, random
from IPython import embed
import json

def run(pre_spike):

    m_n = json.load(open('model_neuron.json','r'))

    m_phi = m_n['phi']

    def get_spike_currents(U, t_post_spike):
        current = 0.0
        if 0.0 <= t_post_spike < 1:
            current += m_n['g_Na']*(m_n['E_Na'] - U)
        if 1.0 <= t_post_spike <= 3:
            current += m_n['g_K']*(m_n['E_K'] - U)
        return current
        
    def phi(U, r_max=m_phi['r_max'], k=m_phi['k'], thresh=m_phi['thresh'], beta=m_phi['beta']):
        return r_max/(1+k*np.exp(beta*(1-U/thresh)))
        
    def phi_prime(U, r_max=m_phi['r_max'], k=m_phi['k'], thresh=m_phi['thresh'], beta=m_phi['beta']):
        exp_term = np.exp(beta*(1-U/thresh))
        return beta*exp_term*k*r_max/(((1+exp_term*k)**2)*thresh)
        
    sim = json.load(open('stdp.json','r'))
    
    if "model_neuron" in sim.keys():
        for key in sim['model_neuron']:
            if key == 'phi':
                m_n["phi"].update(sim["model_neuron"]["phi"])
            else:
                m_n.update(sim[key])

    do_plot = sim.get('plot',[])
    if isinstance(do_plot, unicode):
        if do_plot=='all':
            do_plot = range(sim['reps'])
        else:
            raise Exception('unsupported argument for plot')
        
    do_save = sim.get('save',[])
    if isinstance(do_save, unicode):
        if do_save=='all':
            do_save = range(sim['reps'])
        else:
            raise Exception('unsupported argument for save')

    I_ext_steps = sim.get('I_ext',None)
    if I_ext_steps is None:
        I_ext_steps = np.array([[sim['start'],0.0]])
    else:
        I_ext_steps = np.array(I_ext_steps)
    
    fixed_spikes = sim.get("fixed_spikes",None)
    if fixed_spikes is not None:
        fixed_spikes = np.array(fixed_spikes)
        
    pre_spikes = np.array([pre_spike])
    

    def I_ext(t):
        return I_ext_steps[I_ext_steps[:,0]<=t,1][-1]
        
    def rhs(y, t, t_post_spike,g_E_D,syn_pots_sum):
        # y=[U,V,Vdot]
        dy = np.zeros(3)
        
        dy[0] = -m_n['g_L']*y[0] + m_n['g_D']*(y[1]-y[0]) + I_ext(t)
        if t_post_spike <= 3.0:
            dy[0] = dy[0] + get_spike_currents(y[0],t-last_spike)
            
        dy[1] = -m_n['g_L']*y[1] + m_n['g_S']*(y[0]-y[1]) + g_E_D*(m_n['E_E']-y[1])
        
        dy[2] = -(m_n['g_L']+(m_n['g_S']*m_n['g_L'])/(m_n['g_D']+m_n['g_L'])+g_E_D)*y[2] + (m_n['E_E']-y[1])*syn_pots_sum
        
        return dy

    t_start = sim['start']
    t_end = sim['end']
    dt = sim['dt']
    
    n_steps = int((t_end-t_start)/dt)+1
    
    eta = sim['eta']
    
    curr_t = t_start
    curr_y = np.zeros(3)

    weight = np.zeros(n_steps-1)
    weight[0] = sim['eps']

    ys = np.zeros((n_steps,3))
    curr_write_index = 0
    ys[curr_write_index,:] = curr_y

    last_spike = t_start - 2*m_n['t_ref']
    g_E_D = 0.0
    gs = np.zeros(n_steps)
    gs[0] = g_E_D
    syn_pots_sum = 0.0
    
    while curr_t < t_end - 1.5*dt:
    
        g_E_D = g_E_D + np.sum(np.abs(pre_spikes-curr_t)<1e-10)*weight[curr_write_index]
        g_E_D = g_E_D - dt*g_E_D/sim['tau_s']
        
        gs[curr_write_index+1] = g_E_D
        
        syn_pots_sum = np.sum(np.exp(-(curr_t - pre_spikes[pre_spikes <= curr_t])/sim['tau_s']))
        
        new_y = integrate.odeint(rhs, curr_y,np.array([curr_t,curr_t+dt]),hmax=dt,args=(curr_t-last_spike,g_E_D,syn_pots_sum,))[1,:]
        if curr_t-last_spike > m_n['t_ref']:
            if fixed_spikes is None:
                rate = phi(new_y[0])
                if np.random.rand() <= rate*dt:
                    last_spike = curr_t
            else:
                if np.abs(np.min(curr_t-fixed_spikes)) < 1e-10:
                    last_spike = curr_t
                   
        V_w_star = m_n['g_D']/(m_n['g_D']+m_n['g_L'])*new_y[1]
        dendr_pred = phi(V_w_star)
        h = phi_prime(V_w_star)/phi(V_w_star)
        
        weight[curr_write_index+1] = weight[curr_write_index] + sim['eta']*((last_spike==curr_t) - dt*dendr_pred)*h*new_y[2]
        #embed()
        if weight[curr_write_index+1] < 0.0:
            weight[curr_write_index+1] = 0.0
        
        curr_t = curr_t + dt
        curr_write_index = curr_write_index + 1
        ys[curr_write_index,:] = new_y
        curr_y = new_y
        
    return weight[-1]
    
    if rep in do_plot:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(np.arange(t_start,t_end+dt,dt),ys)
        plt.subplot(3,1,2)
        plt.plot(np.arange(t_start,t_end,dt),np.log10(weight))
        plt.subplot(3,1,3)
        plt.plot(np.arange(t_start,t_end+dt,dt),gs)
        
pres = np.arange(0,101,1)
last_w = np.zeros(pres.shape)
for idx, pre in enumerate(pres):
    print pre
    last_w[idx] = run(pre)
    
    
last_w = last_w - 1e-4
last_w = last_w/np.max(np.abs(last_w))
   
plt.plot(pres,last_w)
plt.xticks(np.arange(0,101,10))
plt.show()