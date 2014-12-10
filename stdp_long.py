from simulation import run
from util import *
import numpy as np
from IPython import embed
from pylab import *
import cPickle
from parallelization import run_tasks
from collections import OrderedDict
import argparse
import os

def do((repetition_i,p)):
	return

	pres = np.arange(10,91,10)

	learn = {}
	learn['eps'] = p['eps']
	learn['eta'] = p['eta']

	res = {}

	t_end = 10000.0

	for idx, pre_spike in enumerate(pres):


		pre_spikes = np.arange(pre_spike,t_end,100.0)
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

		post_spikes = arange(50.0,t_end,100.0)

		my_s['I_ext'] = periodic_current(50.0, 100.0, 0.2, p['I'])

		accum = run(my_s, fixed_spiker(post_spikes), inst_backprop, Accumulator(save, my_s,interval=20), learn=learn)
		res[pre_spike] = accum

	dump(res,p['ident'])

parser = argparse.ArgumentParser(description='Parsing simulation run comment')
parser.add_argument('description', type=str, help='simulation purpose')

nb_descriptors = OrderedDict()
nb_descriptors['simulation file'] = os.path.basename(__file__)
nb_descriptors['description'] = parser.parse_args().description

params = OrderedDict()
params['eta'] = [1e-7,1e-6]
params['eps'] = [1e-1,1e-2,1e-3,1e-4,1e-5]
params['I'] = [0.0, 5.0, 10.0, 20.0, 40.0]

file_prefix = 'stdp_long_with_I'
nb_descriptors['result files prefix'] = file_prefix
runs = construct_params(params,file_prefix)
run_tasks(runs, params, do, nb_descriptors, withmp=False)
