from simulation import run
from util import fixed_spiker, inst_backprop, Accumulator, dendr_spike_det, periodic_current, get_default
import numpy as np
from IPython import embed
from pylab import *
import cPickle
from parallelization import *


def do((repetition_i,p)):

	dc = p['dc']
	width = p['width']
	dendr_spike = p['d_s']

	pres = np.arange(20,81,5)

	learn = get_default("learn")
	learn['eps'] = p['eps']

	res = {}

	t_end = 50000.0

	for idx, pre_spike in enumerate(pres):

		pre_spikes = np.arange(pre_spike,t_end,100.0)
		my_s = {
			'start': 0.0,
			'end': t_end,
			'dt': 0.05,
			'pre_spikes': pre_spikes,
			'I_ext': periodic_current(first=50.0,interval=100,width=width,dcs=[dc,0.0])
			}

		save = ['weight','y','I_ext']

		post_spikes = arange(50.0,t_end,100.0)


		accum = run(my_s, fixed_spiker(post_spikes), inst_backprop, Accumulator(save, my_s,interval=10), learn=learn)
		res[pre_spike] = accum

	cPickle.dump(res, open('{0}.p'.format(p['ident']),'wb'))

reps = 1
dcs = [0.0,20.0]
widths = [0.5,0.2]
epss = [1e-2,2e-2,4e-2,1e-1]
params = constructParams(['dc','width','eps'],[dcs,widths,epss],'long_term_stdp_')
print "running {0} simulations".format(reps*len(params))
run_tasks(reps,params,do,withmp=True)
