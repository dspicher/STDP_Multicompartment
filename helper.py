import numpy as np
from IPython import embed

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
                
    def cleanup(self):
        pass
        
    def add_variable(self,name,val):
        self.res['name'] = val

class PeriodicAccumulator:
    def _get_size(self, key):
        if key=='y':
            return 5
        else:
            return 1

    def __init__(self, keys, interval=1, init_size=1024):
        self.keys = keys
        self.i = interval
        self.j = 0
        self.size = init_size
        self.res = {}
        self.interval = interval
        self.t = np.zeros(init_size, np.float32)
        for key in keys:
            self.res[key] = np.zeros((init_size,self._get_size(key)), np.float32)

    def add(self, curr_t, **vals):
        if np.isclose(self.i, self.interval):
            if self.j == self.size:
                self.t = np.concatenate((self.t,np.zeros(self.t.shape, np.float32)))
                for key in self.keys:
                    self.res[key] = np.vstack((self.res[key],np.zeros(self.res[key].shape, np.float32)))
                self.size = self.size*2
                
            for key in self.keys:
                self.res[key][self.j,:] = np.atleast_2d(vals[key])
            self.t[self.j] = curr_t
            
            self.j += 1
            self.i = 0
        self.i += 1
        
    def cleanup(self):
        self.t = self.t[:self.j]
        for key in self.keys:
            self.res[key] = np.squeeze(self.res[key][:self.j,:])
        
    def add_variable(self,name,val):
        self.res['name'] = val


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

    #create_analysis_notebook(nb_descriptors, params, texts, base_str, "_pre")

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
        pickler_cell_str += name + "_s = " + repr(vals) + "\n"

    names = [k+"_s" for k in ps.keys()   ]
    pickler_cell_str += "\n\n"
    pickler_cell_str += "params = list(product(" + ", ".join(names) + "))"

    pickler_cell_str += "\n\n"
    pickler_cell_str += "data = {tup:get(*tup) for tup in params}"

    cells.append(nbf.new_code_cell(pickler_cell_str))
    
    cells.append(nbf.new_code_cell("from IPython.html.widgets import interact, interactive, fixed\nfrom IPython.html import widgets\nfrom IPython.display import clear_output, display, HTML"))
    
    interact = ""
    interact +="def show_plot(key,"+", ".join(ps.keys())+",y_c,t_min,t_max):\n"
    interact +="    figure(figsize=(12,5))\n"   
    interact +="    p = ("+", ".join(ps.keys())+")\n"
    interact +="    ts = data[p][0].t\n"
    interact +="    mask = np.logical_and(ts>=t_min,ts<=t_max)\n"
    interact +="    if key=='y':\n"
    interact +="        plot(data[p][0].t[mask],data[p][0].res[key][mask,y_c])\n"
    interact +="    else:\n"
    interact +="        plot(data[p][0].t[mask],data[p][0].res[key][mask])\n"
    cells.append(nbf.new_code_cell(interact))
    
    interact =""
    interact += "ts = data[params[0]][0].t\n"
    interact += "i = interact(show_plot,\n"
    interact += "key=widgets.DropdownWidget(description='key',values=['dendr_pred','weight','weight_update', 'PIV', 'y','h']),\n"
    interact += "t_min=(0,int(np.round(ts[-1]))),\n"
    interact += "t_max=(0,int(np.round(ts[-1]))),\n"
    for name, vals in ps.items():
        rep = repr(vals)
        if rep[:6] == "array(":
            rep = rep[6:-1]
        interact += name + "=widgets.RadioButtonsWidget(description=\'" + name + "\',values=" + rep + "),\n"
    interact += "y_c=widgets.RadioButtonsWidget(description='y_c',values=range(5)))\n"
    cells.append(nbf.new_code_cell(interact))
    
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
