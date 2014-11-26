from simulation import run
from util import fixed_spiker
import numpy as np
from IPython import embed
from pylab import *


def robert_plot():
	
	pres = np.arange(0,101,2)
	
	d_ws = np.zeros(pres.shape)
	
	dcs = arange(0.0,1.6,0.3)
	
	width = 0.5
	
	for dc in dcs:
		
		print dc
	
		for idx, pre_spike in enumerate(pres):
			my_s = {
				'start': 0.0,
				'end': 100.0,
				'dt': 0.05,
				'pre_spikes': (pre_spike),
				'I_ext': np.array([[0.0,0.0],[50-width/2,dc],[50+width/2,0.0]])
				}

			ys, weight, gs = run(my_s, fixed_spiker(np.array([50.0])))
			
			d_ws[idx] = (weight[-1] - weight[0])/weight[0]

		plot(d_ws)
		
	legend([str(i) for i in dcs])
	show()

robert_plot()
