from util import fixed_spiker, inst_backprop, periodic_current, dendr_spike_det
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run

def task((repetition_i,p)):

	learn = {}
	learn['eps'] = p['eps_factor']*p['eta']
	learn['eta'] = p['eta']

	res = {}
	first_spike = 1000.0/(2*p['freq'])
	isi = 1000.0/p['freq']
	t_end = 1000.0*100.0/p['freq']

	pre_spikes = np.arange(first_spike + p['delta'],t_end,isi)
	my_s = {
		'start': 0.0,
		'end': t_end,
		'dt': 0.05,
		'pre_spikes': pre_spikes
		}

	post_spikes = np.arange(first_spike,t_end,isi)

	my_s['I_ext'] = periodic_current(first_spike, isi, 0.2, p['I'])

	accs = [PeriodicAccumulator(['weight'], my_s,interval=200), BooleanAccumulator(['spike', 'dendr_spike'])]
	if np.isclose(p['eta'],1e-6) and np.isclose(p['eps_factor'],1e3) and ['dendr_spike']=='dendr_spike_det' and p['I']==5 and p['freq'] >=20:
		accs.append(PeriodicAccumulator(['y','I_ext']))

	accums = run(my_s, fixed_spiker(post_spikes), eval(p['dendr_spike']), accs, learn=learn)

	dump(accums,p['ident'])

params = OrderedDict()
params['eta'] = [1e-6,1e-5]
params['eps_factor'] = [1e3,3e3]
params['freq'] = np.array([2,10,20,40,50])
params['I'] = [0,5,10,20,40]
params['dendr_spike'] = ['inst_backprop','dendr_spike_det']
params['delta'] = np.array([-10,10])

file_prefix = 'sjostrom_frequency'

do(task, params, file_prefix)
