import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, random
from IPython import embed
import json

def run(name):

    model_neuron = json.load(open('model_neuron.json','r'))

    m_phi = model_neuron['phi']

    def get_spike_currents(U, t_post_spike):
        current = 0.0
        if 0.0 <= t_post_spike < 1:
            current += model_neuron['g_Na']*(model_neuron['E_Na'] - U)
        if 1.0 <= t_post_spike <= 3:
            current += model_neuron['g_K']*(model_neuron['E_K'] - U)
        return current
        
    def phi(U, r_max=m_phi['r_max'], k=m_phi['k'], thresh=m_phi['thresh'], beta=m_phi['beta']):
        return r_max/(1+k*np.exp(beta*(1-U/thresh)))
        
    sim = json.load(open(name+'.json','r'))
    
    if "model_neuron" in sim.keys():
        for key in sim['model_neuron']:
            if key == 'phi':
                model_neuron["phi"].update(sim["model_neuron"]["phi"])
            else:
                model_neuron.update(sim[key])

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

    I_ext_steps = np.array(sim['I_ext'])
    
    fixed_spikes = sim.get("fixed_spikes",None)
    if fixed_spikes is not None:
        fixed_spikes = np.array(fixed_spikes)

    def I_ext(t):
        return I_ext_steps[I_ext_steps[:,0]<=t,1][-1]
        
    def rhs(y, t, t_post_spike):
        # y=[U,V]
        dy = np.zeros(2)
        
        dy[0] = -model_neuron['g_L']*y[0] + model_neuron['g_D']*(y[1]-y[0]) + I_ext(t)
        dy[1] = -model_neuron['g_L']*y[1] + model_neuron['g_S']*(y[0]-y[1])
        
        
        if t_post_spike <= 3.0:
            dy[0] = dy[0] + get_spike_currents(y[0],t-last_spike)
        
        return dy

    t_start = sim['start']
    t_end = sim['end']

    dt = sim['dt']
    
    for rep in range(sim['reps']):
        curr_t = t_start
        curr_y = np.zeros(2)

        last_spike = t_start - 2*model_neuron['t_ref']

        n_steps = int((t_end-t_start)/dt)+1

        ys = np.zeros((n_steps,2))
        curr_write_index = 0
        ys[curr_write_index,:] = curr_y

        while curr_t < t_end - dt:
            new_y = integrate.odeint(rhs, curr_y,np.array([curr_t,curr_t+dt]),hmax=dt,args=(curr_t-last_spike,))[1,:]
            if curr_t-last_spike > model_neuron['t_ref']:
                if fixed_spikes is None:
                    rate = phi(new_y[0])
                    if np.random.rand() <= rate*dt:
                        last_spike = curr_t
                else:
                    if np.abs(np.min(curr_t-fixed_spikes)) < 1e-10:
                        last_spike = curr_t
            curr_t = curr_t + dt
            curr_write_index = curr_write_index + 1
            ys[curr_write_index,:] = new_y
            curr_y = new_y
        
        if rep in do_plot:
            plt.figure()
            plt.plot(np.arange(t_start,t_end+dt,dt),ys)
        
        if rep in do_save:
            np.save(open('{0}_rep_{1}'.format(name,rep),'w'),ys)
    
    plt.show()
        
        