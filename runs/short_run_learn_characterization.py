from util import fixed_spiker, inst_backprop, periodic_current, dendr_spike_det, get_all_save_keys
from helper import do, PeriodicAccumulator, BooleanAccumulator, dump
import numpy as np
from IPython import embed
import cPickle
from collections import OrderedDict
from simulation import run
from helper import get_default

def task((repetition_i,p)):

	learn = get_default("learn")
	learn['eps'] = p['eps']

	res = {}

	t_end = 1000.0


	pre_spikes = np.arange(p['pre_spike'],t_end,200.0)
	my_s = {
		'start': 0.0,
		'end': t_end,
		'dt': 0.05,
		'pre_spikes': pre_spikes,
		'I_ext': periodic_current(100.0, 200.0, 0.2, p['I'])
		}

	post_spikes = np.arange(100.0,t_end,200.0)

	accs = [PeriodicAccumulator(get_all_save_keys(), my_s, interval=10), BooleanAccumulator(['spike','dendr_spike'])]

	accums = run(my_s, fixed_spiker(post_spikes), eval(p['dendr_spike']), accs, learn=learn)


	dump(accums,p['ident'])

params = OrderedDict()
params['eps'] = [1e-1,1e-2,1e-3,1e-4]
params['I'] = [0,5,10,20,40]
params['dendr_spike'] = ['inst_backprop','dendr_spike_det']
params['pre_spike'] = 100 + np.array([-50,-20,-10,10,20,50])

file_prefix = 'short_runs_characterization'

do(task, params, file_prefix)
