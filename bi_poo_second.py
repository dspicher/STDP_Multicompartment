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

	post_spikes = np.arange(100.0,t_end,200.0)

	my_s['I_ext'] = periodic_current(100.0, 200.0, 0.5, p['I'])

	accs = [PeriodicAccumulator(['weight'], my_s,interval=200), BooleanAccumulator(['spike', 'dendr_spike'])]

	accums = run(my_s, fixed_spiker(post_spikes), inst_backprop, accs, learn=learn)


	dump(accums,p['ident'])

params = OrderedDict()
params['eta'] = [1e-6]
params['eps'] = [1e-3,3e-3]
params['I'] = np.arange(7,23,3)
params['pre_spike'] = np.array([10,30,50,60,70,75,80,85,90,93,95,96,97,98,99,100,101,102,103,104,105,107,110,115,120,125,130,140,150,170,190])

file_prefix = 'bi_poo_2'

do(task, params, file_prefix)
