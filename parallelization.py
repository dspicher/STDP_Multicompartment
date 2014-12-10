import itertools as it
import multiprocessing as mp
import numpy as np
import datetime
import time
from util import create_analysis_notebook

def run_tasks(runs, params, runTask, nb_descriptors, repetitions = 1, withmp=True):
    '''
    repetitions - integer number repetitions per param
    steps - integer number of steps per task
    params - a list of all param values to be simulated
    withmp - boolean enables multiprocessing, otherwise run serially
    '''
    # below is a generator expression that returns tuples of parameters to be passed
    # into the run_task function. It will have repetitions*len(params) elements.
    run_params = runs['params']
    all_reps_params = ((rep_i, param)
                            for param,rep_i in it.product(run_params, xrange(repetitions)))

    print "running {0} simulations".format(repetitions*len(run_params))
    ts = time.time()
    nb_descriptors['simulation start'] = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    if withmp:
        pool = mp.Pool(mp.cpu_count())
        pool.map(runTask, all_reps_params)
    else:
        for pair in all_reps_params:
            runTask(pair)
    ts = time.time()
    nb_descriptors['simulation end'] = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    create_analysis_notebook(nb_descriptors, params, runs['base_str'])
