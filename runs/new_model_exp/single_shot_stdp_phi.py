
from util import get_all_save_keys, get_periodic_current, get_inst_backprop, get_phi_spiker, get_dendr_spike_det
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump, get_default
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
import matplotlib.pyplot as plt
import time


def task((repetition_i,p)):
	
	res = {}
	
	for delta in np.array([-30,-20,-10,10,20,30]):

		learn = {}
		learn['eta'] = p['eps']*p['eta_factor']
		learn['eps'] = p['eps']
		learn['tau_delta'] = p['tau_delta']

		neuron = get_default("neuron")
		neuron["phi"]["a"] = p["a"]
		neuron["phi"]["log_pref"] = p["log_pref"]

		my_s = {
			'start': 0.0,
			'end': 600.0,
			'dt': 0.05,
			'pre_spikes': np.array([250.0+delta]),
			'I_ext': get_periodic_current(250.0, 1000.0, 0.8, 100.0)
			}

		while True:
			accs = [PeriodicAccumulator(get_all_save_keys(), interval=1), BooleanAccumulator(['spike', 'dendr_spike', 'pre_spike'])]
			if p['d_s'] == 'bp':
				accums = run(my_s, get_phi_spiker(), get_inst_backprop(), accs, seed=int(time.time()), learn=learn, neuron=neuron)
			else:
				accums = run(my_s, get_phi_spiker(), get_dendr_spike_det(thresh=-45.0), accs, seed=int(time.time()), learn=learn, neuron=neuron)
			spks = accums[1].res['spike']
			if spks.shape[0] != 1 or abs(spks[0]-250) > 1.0:
				print p, spks
			else:
				break
		res[delta] = accums
		
	dump(res,p['ident'])

params = OrderedDict()
params['d_s'] = ['bp']
params['eta_factor'] = [1e-3]
params['eps'] = [1e-3]
params['tau_delta'] = [2.0,5.0,10.0,20.0]
params["a"] = [0.25,0.3,0.32,0.35,0.4]
params["log_pref"] = [10.5,11.0,11.5]

file_prefix = 'single_shot_stdp_phi'

do(task, params, file_prefix, prompt=False, withmp=True)
