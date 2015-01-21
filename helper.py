import numpy as np

def dump(res,ident):
    import cPickle
    cPickle.dump(res, open('{0}.p'.format(ident),'wb'))

class BooleanAccumulator:
    def __init__(self, keys):
        self.keys = keys
        self.res = {key:np.array([]) for key in keys}

    def add(self, curr_t, **vals):
        for key in self.keys:
            if vals[key]:
                self.res[key] = np.append(self.res[key], curr_t)

class PeriodicAccumulator:
    def _get_size(self, key):
        if key=='y':
            return 6
        else:
            return 1

    def __init__(self, keys, sim, interval=1):
        self.keys = keys
        self.i = interval
        self.j = 0
        self.res = {}
        self.interval = interval
        eff_steps = np.floor((sim['end']-sim['start'])/(interval*sim['dt']))
        self.t = np.zeros(eff_steps, np.float32)
        for key in keys:
            self.res[key] = np.zeros((eff_steps,self._get_size(key)), np.float32)

    def add(self, curr_t, **vals):
        if np.isclose(self.i, self.interval):
            for key in self.keys:
                self.res[key][self.j,:] = np.atleast_2d(vals[key])

            self.t[self.j] = curr_t
            self.j += 1
            self.i = 0
        self.i += 1


def get_default(params):
    import json
    return json.load(open('default_{0}.json'.format(params),'r'))

def do(func, params, file_prefix, prompt=True, **kwargs):
    from parallelization import run_tasks
    import inspect
    from collections import OrderedDict
    import time, datetime

    texts = OrderedDict()
    if prompt:
        for t in ["Motivation", "Hypothesis"]:
            texts[t] = input("{0}: ".format(t))
    for t in ["Results", "Conclusion"]:
        texts[t] = "<font color='grey'>n/a</font>"

    nb_descriptors = OrderedDict()
    st = inspect.stack()
    nb_descriptors['simulation file'] = st[1][1]
    nb_descriptors['result files prefix'] = file_prefix
    param_counts = map(len,params.values())
    nb_descriptors['# result files'] = '\*'.join(map(str,param_counts)) + ' = ' + str(reduce(lambda x,y:x*y,param_counts))

    runs, base_str = construct_params(params,file_prefix)

    create_analysis_notebook(nb_descriptors, params, texts, base_str, "_pre")

    ts = datetime.datetime.fromtimestamp(time.time())

    nb_descriptors['simulation start'] = ts.strftime('%Y-%m-%d %H:%M:%S')

    run_tasks(runs, func, **kwargs)

    te = datetime.datetime.fromtimestamp(time.time())
    nb_descriptors['simulation end'] = te.strftime('%Y-%m-%d %H:%M:%S')
    nb_descriptors['duration'] = str(datetime.timedelta(seconds=(te-ts).seconds))



    nb_descriptors['repository'], nb_descriptors['revision hash'] = get_git_info()

    create_analysis_notebook(nb_descriptors, params, texts, base_str)

def get_git_info():
    import subprocess, re
    rev_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    repo = subprocess.check_output(['git', 'remote', '-v'])
    repo = re.search('https.*git',repo).group(0).strip()[:-4]
    rev_string = '[' + rev_hash + '](' + repo + '/tree/' + rev_hash + ')'
    return repo, rev_string


def create_analysis_notebook(nb_descriptors, ps, texts, base_str, name_postfix=''):
    from IPython.nbformat import current as nbf

    nb = nbf.new_notebook()

    cells = []

    md_cell = ''
    md_cell += '| Field | Value |\n'
    md_cell += '|-|-|\n'
    md_cell +="\n".join(['| ' + name + ' | ' + des + ' |' for (name,des) in nb_descriptors.items()])
    cells.append(nbf.new_text_cell('markdown', md_cell))

    md_cell = "\n\n".join('### ' + field + "\n" + value for (field, value) in texts.items())
    cells.append(nbf.new_text_cell('markdown', md_cell))

    cells.append(nbf.new_code_cell("%pylab inline\nimport cPickle\nfrom helper import PeriodicAccumulator, BooleanAccumulator\nfrom itertools import product"))

    pickler_cell_str = ""
    pickler_cell_str += "def get(" + ", ".join(ps.keys()) + "):\n"
    pickler_cell_str += "    return cPickle.load(open(\'" + base_str + ".p\'.format(" + ", ".join(ps.keys()) + "),\'rb\'))\n\n\n"



    for name, vals in ps.items():
        pickler_cell_str += name + "s = " + repr(vals) + "\n"

    names = [k+"s" for k in ps.keys()   ]
    pickler_cell_str += "\n\n"
    pickler_cell_str += "params = list(product(" + ", ".join(names) + "))"

    pickler_cell_str += "\n\n"
    pickler_cell_str += "data = {tup:get(*tup) for tup in params}"

    cells.append(nbf.new_code_cell(pickler_cell_str))

    nb['worksheets'].append(nbf.new_worksheet(cells=cells))

    fname = nb_descriptors['simulation file'][:-3] + "_analysis" + name_postfix + ".ipynb"

    with open(fname, 'w') as f:
        nbf.write(nb, f, 'ipynb')

def construct_params(params, prefix=''):
    from itertools import product
    from operator import add

    ids =tuple(params.keys())
    values = tuple(params.values())

    if prefix.endswith("_"):
        prefix = prefix[:-1]

    base_str = prefix + reduce(add, ['_{0}_{{{1}}}'.format(ids[i],i) for i in range(len(ids))])

    combinations = product(*values)
    concat_params = []
    for comb in combinations:
        curr = {id:val for (id,val) in zip(ids,comb)}
        curr['ident'] = base_str.format(*comb)
        concat_params.append(curr)

    return concat_params, base_str
