import itertools as it
import multiprocessing as mp
import numpy as np

def run_tasks(repetitions, params, runTask, withmp=True):
    '''
    repetitions - integer number repetitions per param
    steps - integer number of steps per task
    params - a list of all param values to be simulated
    withmp - boolean enables multiprocessing, otherwise run serially
    '''
    # below is a generator expression that returns tuples of parameters to be passed
    # into the run_task function. It will have repetitions*len(params) elements.
    all_reps_params = ((rep_i, param)
                            for param,rep_i in it.product(params, xrange(repetitions)))
    if withmp:
        pool = mp.Pool(mp.cpu_count())
        pool.map(runTask, all_reps_params)
    else:
        for pair in all_reps_params:
            runTask(pair)



def constructParams(ids,values,prefix=''):
    combinations = it.product(*values)
    params = []
    for comb in combinations:
        curr = {}
        ident = prefix
        for idx in range(len(comb)):
            curr[ids[idx]] = comb[idx]
            ident = ident+'{0}_{1}_'.format(ids[idx],comb[idx])
        curr['ident'] = ident
        params.append(curr)
    return params
