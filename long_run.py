from simulation import run
from util import fixed_spiker, inst_backprop, Accumulator, dendr_spike_det, periodic_current
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

	res = {}

	t_end = 100.0

	for idx, pre_spike in enumerate(pres):

		pre_spikes = np.arange(pre_spike,t_end,100.0)
		my_s = {
			'start': 0.0,
			'end': t_end,
			'dt': 0.05,
			'pre_spikes': np.array([pre_spike]),
			'I_ext': periodic_current(first=pre_spike,interval=100,width=width,dcs=[dc,0.0])
			}

		save = ['weight']

		post_spikes = arange(50.0,t_end,100.0)

		if dendr_spike == 'backprop':
			d_s = inst_backprop
		elif dendr_spike == 'dendr_det':
			d_s = dendr_spike_det()
		else:
			raise Exception()

		accum = run(my_s, fixed_spiker(post_spikes), d_s, Accumulator(save, my_s,interval=20))
		res[pre_spike] = accum

	cPickle.dump(res, open('{0}.p'.format(p['ident']),'wb'))

reps = 1
dcs = [0.0,10.0,20.0,40.0]
widths = [0.5,0.2,0.1]
d_s = ['backprop','dendr_det']
params = constructParams(['dc','width','d_s'],[dcs,widths,d_s],'long_term_stdp')
print "running {0} simulations".format(reps*len(params))
run_tasks(reps,params,do,withmp=True)
