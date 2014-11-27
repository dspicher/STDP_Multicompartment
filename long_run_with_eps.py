from simulation import run
from util import fixed_spiker, inst_backprop, Accumulator, dendr_spike_det, periodic_current, get_default
import numpy as np
from IPython import embed
from pylab import *
import cPickle
from parallelization import *


def do((repetition_i,p)):

	pres = np.arange(20,81,10)

	learn = {}
	learn['eps'] = p['eps']
	learn['eta'] = p['eta']

	res = {}

	t_end = 10000.0

	for idx, pre_spike in enumerate(pres):

		print pre_spikes

		pre_spikes = np.arange(pre_spike,t_end,100.0)
		my_s = {
			'start': 0.0,
			'end': t_end,
			'dt': 0.05,
			'pre_spikes': pre_spikes,
			'I_ext': lambda t: 0.0
			}

		vals = {'g':g_E_D,
				'syn_pots_sum':syn_pots_sum,
				'y':y,
				'spike':float(does_spike),
				'V_w_star':V_w_star,
				'dendr_pred':dendr_pred,
				'h':h,
				'dendr_spike':float(dendr_spike),
				'weight':weight,
				'weight_update':weight_update,
				'I_ext':I_ext(curr_t - dt)}

		save = vals.keys()

		post_spikes = arange(50.0,t_end,100.0)


		accum = run(my_s, fixed_spiker(post_spikes), inst_backprop, Accumulator(save, my_s,interval=20), learn=learn)
		res[pre_spike] = accum

	cPickle.dump(res, open('{0}.p'.format(p['ident']),'wb'))

reps = 1
etas = [1e-5,1e-4,1e-3,1e-2]
epss = [1e-3,1e-4,1e-2,1e-1]
params = constructParams(['eta','eps'],[etas,epss],'long_term_stdp_')
print "running {0} simulations".format(reps*len(params))
run_tasks(reps,params,do,withmp=True)
