from __future__ import division
import numpy as np
import pandas as pd
import os
import yaml
import re
import glob
import cPickle as pickle
import itertools
from StringIO import StringIO
import parameter_prediction.util.score_timit_model as sc_timit

def job_results_available(job_dir):
    model_file_name = os.path.join(job_dir, "models", "finetune_all.pkl")
    return os.path.exists(model_file_name)

def job_dirs_with_results(root):
    dirs = []
    for dir in os.listdir(root):
        full_dir = os.path.join(root, dir)
        if job_results_available(full_dir):
            dirs.append(full_dir)
    return dirs

def load_params(param_file_name):
    with open(param_file_name) as param_file:
        return yaml.load(param_file)

def get_params(job_dir):
    n_params_total = 0
    n_params_actual = 0
    n_columns = None

    def load_from_job_dir(fname):
        return load_params(os.path.join(job_dir, fname))

    job_id = load_from_job_dir("launcher_params.yaml")['job_id']

    for layer_params in map(load_from_job_dir, ["pretrain_layer1_params.yaml", "pretrain_layer2_params.yaml"]):
        n_vis = layer_params['n_vis']

        if n_columns is not None:
            assert n_columns == len(layer_params['columns'])
        n_columns = len(layer_params['columns'])

        for column in layer_params['columns']:
            n_params_total += layer_params['n_vis'] * column['n_hid']
            n_params_actual += column['n_atoms'] * column['n_hid']

    params = load_from_job_dir("finetune_all_params.yaml")
    params['n_params_total'] = n_params_total
    params['n_params_actual'] = n_params_actual
    params['params_prop'] = n_params_actual / n_params_total
    params['n_columns'] = n_columns
    params['job_id'] = job_id

    return params

def load_model(model_path):
    with open(model_path) as model_file:
        return pickle.load(model_file)

def get_model(job_dir):
    model_names = ["finetune_all.pkl"]
    model_paths = [os.path.join(job_dir, "models", name) for name in model_names]
    models = [load_model(m) for m in model_paths]
    return models[0]

def get_model_and_params(job_dir):
    return [get_model(job_dir), get_params(job_dir)]

if __name__ == "__main__":
    root = "scratch_space_COV"
    model_params = itertools.imap(get_model_and_params, job_dirs_with_results(root))

    # make sure the order here matches the order in which the lines are constructed
    keys = [
        ]

    values = [
        "params_prop",
        "n_columns",
        "job_id",
        "PER",
        ]

    report_lines = []
    report_lines.append(",".join(keys + values))
    for i, (model, params) in enumerate(model_params):
        print i

        line = ",".join(map(str, [
            params['params_prop'],
            params['n_columns'],
            params['job_id'],
            sc_timit.get_PER(model)
            ]))
        report_lines.append(line)

    df = pd.read_csv(StringIO("\n".join(report_lines)))

    df.to_csv("report_updated.csv")
