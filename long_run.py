from simulation import run
from util import fixed_spiker, inst_backprop, Accumulator, dendr_spike_det, periodic_current
import numpy as np
from IPython import embed
from pylab import *
import cPickle


def do(dc, width, dendr_spike):

	pres = np.arange(0,101,2)

	res = {}

	t_end = 10000.0

	for idx, pre_spike in enumerate(pres):
		my_s = {
			'start': 0.0,
			'end': t_end,
			'dt': 0.05,
			'pre_spikes': np.array([pre_spike]),
			'I_ext': periodic_current(first=pre_spike,interval=100,width=width,dcs=[dc,0.0])
			}

		save = ['y','weight','dendr_spike','spike']

		post_spikes = arange(50,100.0,t_end)

		if dendr_spike == 'backprop':
			d_s = inst_backprop
		elif dendr_spike == 'dendr_det':
			d_s = dendr_spike_det()
		else:
			raise Exception()

		accum = run(my_s, fixed_spiker(post_spikes), d_s, Accumulator(save, my_s))
		res[pre_spike] = accum

	cPickle.dump(res, open('long_term_stdp_dc_{0}_width_{1}_{2}.p'.format(dc,width,dendr_spike),'wb'))

for dc in [0.0,10.0,20.0,40.0]:
	for width in [0.5,0.2,0.1]:
		for d_s in ['backprop','dendr_det']:
			do(dc,width,d_s)
