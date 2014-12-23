from util import fixed_spiker, inst_backprop, periodic_current, dendr_spike_det
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run

def task((repetition_i,p)):

	learn = {}
	learn['eta'] = p['eta']
	learn['eps'] = 1e3*p['eta']

	res = {}
	first_spike = 1000.0/(2*p['freq'])
	isi = 1000.0/p['freq']
	t_end = 1000.0*100.0/p['freq']

	pre_spikes = np.arange(first_spike + p['delta'],t_end,isi)
	my_s = {
		'start': 0.0,
		'end': t_end,
		'dt': 0.05,
		'pre_spikes': pre_spikes,
		'I_ext': lambda t: 0.0
		}

	post_spikes = np.arange(first_spike,t_end,isi)

	accs = [PeriodicAccumulator(['weight'], my_s,interval=20), BooleanAccumulator(['spike', 'dendr_spike'])]

	accums = run(my_s, fixed_spiker(post_spikes), eval(p['dendr_spike']), accs, learn=learn)

	dump(accums,p['ident'])

params = OrderedDict()
params['eta'] = [1e-6,1e-5]
params['freq'] = np.array([2,10,20,40,50])
params['dendr_spike'] = ['inst_backprop']
params['delta'] = np.array([-20,-10,-5,5,10,20])

file_prefix = 'sjostrom_frequency'

do(task, params, file_prefix)
