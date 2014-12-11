from util import fixed_spiker, inst_backprop, periodic_current, dendr_spike_det
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run

def task((repetition_i,p)):

	learn = {}
	learn['eps'] = p['eps']
	learn['eta'] = p['eta']

	res = {}

	t_end = 20000.0


	pre_spikes = np.arange(p['pre_spike'],t_end,200.0)
	my_s = {
		'start': 0.0,
		'end': t_end,
		'dt': 0.05,
		'pre_spikes': pre_spikes,
		'I_ext': lambda t: 0.0
		}

	vals = {'g':0,
			'syn_pots_sum':0,
			'V_w_star':0,
			'weight':0,
			'y':0}

	save = vals.keys()

	post_spikes = np.arange(100.0,t_end,200.0)

	my_s['I_ext'] = periodic_current(100.0, 200.0, 0.5, p['I'])

	accs = [PeriodicAccumulator(save, my_s,interval=20), BooleanAccumulator(['spike', 'dendr_spike'])]

	accums = run(my_s, fixed_spiker(post_spikes), eval(p['dendr_spike']), accs, learn=learn)


	dump(accums,p['ident'])

params = OrderedDict()
params['eta'] = [1e-6]
params['eps'] = [1e-3,3e-3,1e-2]
params['I'] = [0.0, 10.0, 40.0]
params['dendr_spike'] = ['inst_backprop', 'dendr_spike_det']
params['pre_spike'] = np.array([10,30,50,60,70,75,80,85,90,95,98,100,102,105,110,115,120,125,130,140,150,170,190])

file_prefix = 'stdp_characterization2'

do(task, params, file_prefix)
