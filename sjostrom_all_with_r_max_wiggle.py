
from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_fixed_spiker, get_dendr_spike_det_dyn_ref
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time


def task((repetition_i,p)):

    learn = {}
    learn['eta'] = 1e-7
    learn['eps'] = 1e-3
    learn['tau_delta'] = 2.0
    
    ident = "sj_find_no_ltp_alpha_{0}_beta_{1}_g_L_{2}".format(p["alpha"], p["beta"], p["g_L"])
    no_ltp_hist = cPickle.load(open(ident+".p",'rb'))["hist"]
    if len(no_ltp_hist) <= 2:
        print "had no result for {0}".format(p["ident"])
        dump([],p['ident'])
        return
    
    r_max = no_ltp_hist[-1][0]
    
    runs = []
    dws = []
    
    freqs = np.array([1.0,10.0,20.0,40.0,50.0])
    deltas = np.array([-10.0,10.0])
    
    n_spikes = 40.0
    check_interval = 5
    check_spikes = np.arange(check_interval,n_spikes+1,check_interval)
    
    for r in [0.9*r_max, r_max, 1.1*r_max]:
        
        dw_curves = np.zeros((freqs.shape[0],deltas.shape[0], check_spikes.shape[0]))
    
        for freq_idx, freq in enumerate(freqs):
            
            for delta_idx, delta in enumerate(deltas):

                neuron = get_default("neuron")
                neuron["phi"]['r_max'] = r
                neuron["phi"]['alpha'] = p["alpha"]
                neuron["phi"]['beta'] = p["beta"]
                neuron["g_L"] = p["g_L"]

                first_spike = 1000.0/(2*freq)
                isi = 1000.0/freq
                t_end = 1000.0*n_spikes/freq
                
                spikes = np.arange(first_spike, t_end, isi)
                pre_spikes = spikes + delta
                
                my_s = {
                    'start': 0.0,
                    'end': t_end,
                    'dt': 0.05,
                    'pre_spikes': pre_spikes,
                    'I_ext': lambda t: 0.0
                    }
                    
                seed = int(int(time.time()*1e8)%1e9)
                accs = [PeriodicAccumulator(['weight'], interval=10)]
                accums = run(my_s, get_fixed_spiker(spikes), get_dendr_spike_det_dyn_ref(-50.0,10.0,100.0), accs, seed=seed, learn=learn, neuron=neuron)

                runs.append((r, freq, delta, seed, accums[0].t, accums[0].res['weight']))
                
                for check_idx, check_spike in enumerate(check_spikes):
                    t = spikes[check_spike-1] + 0.9*isi
                    match_idx = np.argsort(np.abs(accums[0].t-t))[0]
                    dw_curves[freq_idx, delta_idx,check_idx] = accums[0].res['weight'][match_idx]/accums[0].res['weight'][0]
                
        dws.append(dw_curves)
                
    dump((dws, runs),p['ident'])

params = OrderedDict()
params['alpha'] = np.linspace(-60,-50,6)
params["beta"] = [0.1,0.2]
params["g_L"] = [0.05, 0.03]

file_prefix = 'sjostrom_curves'

do(task, params, file_prefix, prompt=False, withmp=True)

