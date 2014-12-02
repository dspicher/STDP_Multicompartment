from simulation import run
from util import *
import numpy as np
from IPython import embed
from pylab import *
import cPickle
from parallelization import run_tasks


def do((repetition_i,p)):

	pres = np.arange(20,81,5)

	learn = {}
	learn['eps'] = p['eps']
	learn['eta'] = p['eta']

	neuron = get_default("neuron")
	neuron['delta_factor'] = p['factor']

	res = {}

	t_end = 5000.0

	for idx, pre_spike in enumerate(pres):

		print pre_spike

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
				'dendr_pred':0,
				'y':0,
				'h':0,
				'weight':0}

		save = vals.keys()

		post_spikes = arange(50.0,t_end,100.0)


		accum = run(my_s, fixed_spiker(post_spikes), inst_backprop, Accumulator(save, my_s,interval=10), neuron=neuron, learn=learn)
		res[pre_spike] = accum

	dump(res,p['ident'])

reps = 1
etas = [5e-4,1e-3]
epss = [5e-3,2e-2]
factors = [1, 2, 5, 10, 15, 20, 30, 40]
params = construct_params(['eta','eps','factor'],[etas,epss,factors],'stdp_stabilization_factor_')
print "running {0} simulations".format(reps*len(params))
run_tasks(reps,params,do,withmp=True)
