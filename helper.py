import numpy as np
from IPython import embed


def dump(res, ident):
    import cPickle
    cPickle.dump(res, open('{0}.p'.format(ident), 'wb'))


class BooleanAccumulator:

    def __init__(self, keys):
        self.keys = keys
        self.res = {key: np.array([]) for key in keys}

    def add(self, curr_t, **vals):
        for key in self.keys:
            if vals[key]:
                self.res[key] = np.append(self.res[key], curr_t)

    def prepare_arrays(self, n_syn):
        pass

    def cleanup(self):
        pass

    def add_variable(self, name, val):
        self.res[name] = val


class PeriodicAccumulator:

    def _get_size(self, key):
        if key == 'y':
            if self.y_keep is not None:
                return self.y_keep
            return 3 + 2 * self.n_syn
        elif key in ['g_E_Ds', 'syn_pots_sums', 'PIVs', 'pos_PIVs', 'neg_PIVs', 'weights', 'weight_updates', 'deltas', 'pre_spikes', 'dendr_pred']:
            return self.n_syn
        else:
            return 1

    def __init__(self, keys, interval=1, init_size=1024, y_keep=None):
        self.keys = keys
        self.init_size = init_size
        self.i = interval
        self.j = 0
        self.size = init_size
        self.interval = interval
        self.t = np.zeros(init_size, np.float32)
        self.y_keep = y_keep

    def prepare_arrays(self, n_syn=1):
        self.n_syn = n_syn
        self.res = {}
        for key in self.keys:
            self.res[key] = np.zeros((self.init_size, self._get_size(key)), np.float32)

    def add(self, curr_t, **vals):
        if np.isclose(self.i, self.interval):
            if self.j == self.size:
                self.t = np.concatenate((self.t, np.zeros(self.t.shape, np.float32)))
                for key in self.keys:
                    self.res[key] = np.vstack(
                        (self.res[key], np.zeros(self.res[key].shape, np.float32)))
                self.size = self.size * 2

            for key in self.keys:
                if key == 'y' and self.y_keep is not None:
                    self.res[key][self.j, :] = np.atleast_2d(vals[key][:self.y_keep])
                else:
                    self.res[key][self.j, :] = np.atleast_2d(vals[key])
            self.t[self.j] = curr_t

            self.j += 1
            self.i = 0
        self.i += 1

    def cleanup(self):
        self.t = self.t[:self.j]
        for key in self.keys:
            self.res[key] = np.squeeze(self.res[key][:self.j, :])

    def add_variable(self, name, val):
        self.res[name] = val


def get_default(params):
    import json
    return json.load(open('./default/default_{0}.json'.format(params), 'r'))


def do(func, params, file_prefix, create_notebooks=True, **kwargs):
    from parallelization import run_tasks
    import inspect
    from collections import OrderedDict
    import time
    import datetime

    runs, base_str = construct_params(params, file_prefix)

    if create_notebooks:
        nb_descriptors = OrderedDict()
        st = inspect.stack()
        nb_descriptors['simulation file'] = st[1][1]
        nb_descriptors['result files prefix'] = file_prefix
        param_counts = map(len, params.values())
        nb_descriptors['# result files'] = '\*'.join(map(str, param_counts)) + \
            ' = ' + str(reduce(lambda x, y: x * y, param_counts, ""))

        create_analysis_notebook(nb_descriptors, params, base_str, "_pre")

        ts = datetime.datetime.fromtimestamp(time.time())

        nb_descriptors['simulation start'] = ts.strftime('%Y-%m-%d %H:%M:%S')

    run_tasks(runs, func, **kwargs)

    if create_notebooks:

        te = datetime.datetime.fromtimestamp(time.time())
        nb_descriptors['simulation end'] = te.strftime('%Y-%m-%d %H:%M:%S')
        nb_descriptors['duration'] = str(datetime.timedelta(seconds=(te - ts).seconds))

        nb_descriptors['repository'], nb_descriptors['revision hash'] = get_git_info()
        create_analysis_notebook(nb_descriptors, params, base_str)


def get_git_info():
    import subprocess
    import re
    rev_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    repo = subprocess.check_output(['git', 'remote', '-v'])
    repo = re.search('https.*git', repo).group(0).strip()[:-4]
    rev_string = '[' + rev_hash + '](' + repo + '/tree/' + rev_hash + ')'
    return repo, rev_string


def create_analysis_notebook(nb_descriptors, ps, base_str, name_postfix=''):
    import nbformat as nbf
    import os

    nb = nbf.v4.new_notebook()

    cells = []

    md_cell = ''
    md_cell += '| Field | Value |\n'
    md_cell += '|-|-|\n'
    md_cell += "\n".join(['| ' + name + ' | ' + des +
                          ' |' for (name, des) in nb_descriptors.items()])
    cells.append(nbf.v4.new_markdown_cell(md_cell))

    cells.append(nbf.v4.new_code_cell(
    "import os,sys,inspect\ncurrentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))\nparentdir = os.path.dirname(currentdir)\nsys.path.insert(0,parentdir)"))

    cells.append(nbf.v4.new_code_cell(
        "%pylab inline\nimport cPickle\nfrom helper import PeriodicAccumulator, BooleanAccumulator\nfrom itertools import product\nplt.style.use('ggplot')"))

    pickler_cell_str = ""
    pickler_cell_str += "def get(" + ", ".join(ps.keys()) + "):\n"
    pickler_cell_str += "    return cPickle.load(open(\'" + base_str + \
        ".p\'.format(" + ", ".join(ps.keys()) + "),\'rb\'))\n\n\n"

    for name, vals in ps.items():
        pickler_cell_str += name + "_s = [str(a) for a in " + repr(vals) + "]\n"

    names = [k + "_s" for k in ps.keys()]
    pickler_cell_str += "\n\n"
    pickler_cell_str += "params = list(product(" + ", ".join(names) + "))"

    pickler_cell_str += "\n\n"
    pickler_cell_str += "data = {tup:get(*tup) for tup in params}"

    cells.append(nbf.v4.new_code_cell(pickler_cell_str))

    cells.append(nbf.v4.new_code_cell(
        "from ipywidgets import interact, ToggleButtons"))

    interact = ""
    interact += "def show_plot(key," + ", ".join(ps.keys()) + ",y_c,t_min,t_max):\n"
    interact += "    figure(figsize=(12,5))\n"
    interact += "    p = (" + ", ".join(ps.keys()) + ")\n"
    interact += "    curr = data[p][1][0]\n"
    interact += "    ts = curr.t\n"
    interact += "    mask = np.logical_and(ts>=t_min,ts<=t_max)\n"
    interact += "    if key=='y':\n"
    interact += "        plot(curr.t[mask],curr.res[key][mask,:int(y_c)+1])\n"
    interact += "    else:\n"
    interact += "        plot(curr.t[mask],curr.res[key][mask])\n"
    cells.append(nbf.v4.new_code_cell(interact))


    interact = ""
    interact += "ts = data[params[0]][1][0].t\n"
    interact += "i = interact(show_plot,\n"
    interact += "key=ToggleButtons(description='key',options=['dendr_pred','weights','weight_updates', 'PIVs', 'y','h']),\n"
    interact += "t_min=(0,int(np.round(ts[-1]))),\n"
    interact += "t_max=(0,int(np.round(ts[-1]))),\n"
    for name, vals in ps.items():
        rep = repr(vals)
        if rep[:6] == "array(":
            rep = rep[6:-1]
        interact += name + \
            "=ToggleButtons(description=\'" + name + "\',options=" + name + "_s" + "),\n"
    interact += "y_c=ToggleButtons(description='y_c',options=[str(a) for a in range(5)]))\n"
    cells.append(nbf.v4.new_code_cell(interact))

    nb['cells'] = cells

    sim_file = nb_descriptors['simulation file'][:-3]
    fname = sim_file + "_analysis" + name_postfix + ".ipynb"

    if not os.path.exists(sim_file):
        os.makedirs(sim_file)

    with open(sim_file + '/' + fname, 'w') as f:
        nbf.write(nb, f)


def construct_params(params, prefix=''):
    from itertools import product
    from operator import add

    ids = tuple(params.keys())
    values = tuple(params.values())

    if prefix.endswith("_"):
        prefix = prefix[:-1]

    base_str = prefix + reduce(add, ['_{0}_{{{1}}}'.format(ids[i], i) for i in range(len(ids))], "")

    combinations = product(*values)
    concat_params = []
    for comb in combinations:
        curr = {id: val for (id, val) in zip(ids, comb)}
        curr['ident'] = base_str.format(*comb)
        concat_params.append(curr)

    return concat_params, base_str
