import json
from model import phi, phi_prime
import numpy as np
from IPython import embed
from operator import add
import itertools as it
import cPickle

def get_default(params):
    return json.load(open('default_{0}.json'.format(params),'r'))

def get_all_save_keys():
    return ['g',
            'syn_pots_sum',
            'y',
            'spike',
            'V_w_star',
            'dendr_pred',
            'h',
            'dendr_spike',
            'weight',
            'weight_update',
            'I_ext']

def fixed_spiker(spikes):
    return lambda curr_t, dt, **kwargs: np.min(np.abs(curr_t-spikes)) < dt/2

def phi_spiker(phi_params=None):
    if phi_params is None:
        phi_params = get_default_phi()

    return lambda y, dt, **kwargs: phi(y[0])*dt <= np.random.rand()

def inst_backprop(curr_t, last_spike, **kwargs):
    return curr_t==last_spike

def dendr_spike_det(thresh=1.0, tau=4.0):
    return lambda y, curr_t, last_spike_dendr, **kwargs: y[1] > thresh and (curr_t-last_spike_dendr > tau)

def step_current(steps):
    return lambda t: steps[steps[:,0]<=t,1][-1]

def periodic_current(first, interval, width, dc_on, dc_off=0.0):
    def I_ext(t):
        if np.abs(t%interval - first) <= width/2 or np.isclose(first - t%interval, width/2):
            if np.isclose(t%interval,first+width/2):
                return dc_off
            else:
                return dc_on
        else:
            return dc_off
    return I_ext

def get_git_revision_hash():
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def create_analysis_notebook(nb_descriptors, ps, base_str):
    from IPython.nbformat import current as nbf
    nb_descriptors['git revision hash'] = get_git_revision_hash().strip()

    nb = nbf.new_notebook()

    cells = []

    md_cell = ''
    md_cell += '| Field | Value |\n'
    md_cell += '|-|-|\n'
    md_cell +="\n".join(['| ' + name + ' | ' + des + ' |' for (name,des) in nb_descriptors.items()])

    cells.append(nbf.new_text_cell('markdown',md_cell))

    cells.append(nbf.new_code_cell("%pylab inline\nimport cPickle\nfrom util import Accumulator\nfrom itertools import product"))

    pickler_cell_str = ""
    pickler_cell_str += "def get(" + ", ".join(ps.keys()) + "):\n"
    pickler_cell_str += "    return cPickle.load(open(\'" + base_str + ".p\'.format(" + ", ".join(ps.keys()) + "),\'rb\'))\n\n\n"



    for name, vals in ps.items():
        pickler_cell_str += name + "s = " + str(vals) + "\n"

    names = [k+"s" for k in ps.keys()   ]
    pickler_cell_str += "\n\n"
    pickler_cell_str += "params = list(product(" + ", ".join(names) + "))"

    pickler_cell_str += "\n\n"
    pickler_cell_str += "data = {tup:get(*tup) for tup in params}"

    cells.append(nbf.new_code_cell(pickler_cell_str))

    nb['worksheets'].append(nbf.new_worksheet(cells=cells))

    fname = nb_descriptors['simulation file'][:-3] + "_analysis.ipynb"

    with open(fname, 'w') as f:
        nbf.write(nb, f, 'ipynb')


def construct_params(params, prefix=''):
    ids =tuple(params.keys())
    values = tuple(params.values())

    if prefix.endswith("_"):
        prefix = prefix[:-1]

    base_str = prefix + reduce(add, ['_{0}_{{{1}}}'.format(ids[i],i) for i in range(len(ids))])

    combinations = it.product(*values)
    concat_params = []
    for comb in combinations:
        curr = {id:val for (id,val) in zip(ids,comb)}
        curr['ident'] = base_str.format(*comb)
        concat_params.append(curr)

    return {'params': concat_params, 'base_str':base_str}

def dump(res,ident):
    cPickle.dump(res, open('{0}.p'.format(ident),'wb'))

class Accumulator():
    def _get_size(self, key):
        if key=='y':
            return 3
        else:
            return 1

    def __init__(self, keys, sim, interval=1):
        self.keys = keys
        self.i = interval
        self.j = 0
        self.res = {}
        self.interval = interval
        eff_steps = np.arange(sim['start'], sim['end']+sim['dt'], sim['dt']*interval).shape[0]
        self.t = np.zeros(eff_steps)
        for key in keys:
            self.res[key] = np.zeros((eff_steps,self._get_size(key)))

    def add(self, curr_t, **vals):
        if np.abs(self.i-self.interval) < 1e-10:
            for key in self.keys:
                self.res[key][self.j,:] = np.atleast_2d(vals[key])

            self.t[self.j] = curr_t
            self.j += 1
            self.i = 0
        self.i += 1
