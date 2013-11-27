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
    n_epochs = [20,20,100]
    n_epochs_dict = 2
    n_atoms_dict = 1024
    n_hid = 1024

    n_columns = [1, 5, 10]
    n_points_prop = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
        delayed(run_job)(job_id, [task], task_factory=job_manager.ClusterTask)
        for job_id, task in enumerate(tasks))

