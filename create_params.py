import json

model_neuron = {}

model_neuron['g_K'] = 1.5
model_neuron['E_K'] = -4.0/3

model_neuron['g_L'] = 1.0/15
model_neuron['E_L'] = 0.0

model_neuron['g_Na'] = 2.0
model_neuron['E_Na'] = 8.0+2.0/3

model_neuron['E_I'] = -1.0/3
model_neuron['E_E'] = 4.0+2.0/3

model_neuron['g_D'] = 2.0
model_neuron['g_S'] = 0.5

model_neuron['phi'] = {}
model_neuron['phi']['r_max'] = 0.3
model_neuron['phi']['k'] = 1.0
model_neuron['phi']['thresh'] = 1.0
model_neuron['phi']['beta'] = 6.0

model_neuron['t_ref'] = 4.0

json.dump(model_neuron,open('model_neuron.json','w'))