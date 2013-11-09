import numpy as np
import os
import itertools
from pprint import pprint

import job_manager
import config

from joblib import Parallel, delayed


def run_job(job_id, tasks, task_factory):
    pprint(tasks)

    job = job_manager.Job(
        job_id=job_id,
        base_dir=config.base_dir,
        params=config.global_template_params,
        template_dir=config.template_dir,
        tasks=tasks,
        task_factory=task_factory)
    
    for task in job.tasks():
        task.configure()
        task.launch()

if __name__ == "__main__":
    n_jobs = 1

    # set up parameters for different jobs
    n_epochs = [50,50,100]
    n_epochs_dict = 50
    n_atoms_dict = 1000
    n_hid = 500

    n_columns = [1, 5, 10]
    #n_points_prop = [p for p in map(float, np.logspace(np.log10(20.0/784), np.log10(1.0), 10))]
    n_points_prop = [p for p in map(float, np.linspace(20.0/784, 1.0, 10))]

    # generate a list of tasks
    # use itertools.product(...) to generate grids of parameters
    tasks = []
    for nc,p in itertools.product(n_columns, n_points_prop):
        tasks.append(config.get_job(
            n_epochs=n_epochs,
            n_hid=n_hid,
            n_columns=nc,
            prop=p,
            n_epochs_dict=n_epochs_dict,
            n_atoms_dict=n_atoms_dict,
            ))

    print "NUMBER OF TASKS:", len(tasks)

    # launch each task as a job
    Parallel(n_jobs=n_jobs)(
        # change LocalTask to ClusterTask to submit jobs to the cluster
        delayed(run_job)(job_id, [task], task_factory=job_manager.LocalTask)
        for job_id, task in enumerate(tasks))

