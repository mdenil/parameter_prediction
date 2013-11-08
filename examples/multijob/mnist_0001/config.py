import numpy as np
import os
import sys

base_dir = "scratch_space"
template_dir = "templates"

# This dictionary is available to all templates.
# You can add parameters here.
global_template_params = {
    }

#def _n_atoms_per_column(n_hid, n_columns, prop):
#    return int(_n_hid_per_column(n_hid, n_columns) * prop)

def _n_atoms_per_column(n_vis, prop):
    return int(n_vis * prop)

def _n_hid_per_column(n_hid, n_columns):
    return int(n_hid / n_columns)

def _n_hid_total(n_hid, n_columns):
    return _n_hid_per_column(n_hid, n_columns) * n_columns

def get_job(
        n_hid,
        n_columns,
        prop,
        n_epochs,
        n_atoms_dict,
        n_epochs_dict,
        ):
    return {
        "templates": [
            # model config files
            {
                "target": "pretrain_layer1.yaml",
                "params_target": "pretrain_layer1_params.yaml",
                "src": "SE_AE/pretrain_layer1.yaml",
                "params": {
                    "n_epochs": n_epochs[0],
                    "n_vis": 784,
                    "columns": [
                        {
                            "n_atoms": _n_atoms_per_column(784, prop),
                            "n_hid": _n_hid_per_column(n_hid, n_columns)
                        },
                    ] * n_columns,
                },
            },

            {
                "target": "learn_dict_layer2.yaml",
                "params_target": "learn_dict_layer2_params.yaml",
                "src": "SE_AE/learn_dict_layer2.yaml",
                "params": {
                    "n_vis": _n_hid_total(n_hid, n_columns),
                    "n_atoms": n_atoms_dict,
                    "n_epochs": n_epochs_dict,
                },
            },
            
            {
                "target": "pretrain_layer2.yaml",
                "params_target": "pretrain_layer2_params.yaml",
                "src": "SE_AE/pretrain_layer2.yaml",
                "params": {
                    "n_epochs": n_epochs[1],
                    "n_vis": _n_hid_total(n_hid, n_columns),
                    "columns": [
                        {
                            "n_atoms": _n_atoms_per_column(_n_hid_total(n_hid, n_columns), prop),
                            "n_hid": _n_hid_per_column(n_hid, n_columns)
                        },
                    ] * n_columns,
                },
            },
            
            {
                "target": "finetune_all.yaml",
                "params_target": "finetune_all_params.yaml",
                "src": "SE_AE/finetune_all.yaml",
                "params": {
                    "n_epochs": n_epochs[2],
                    "n_vis": 784,
                },
            },
            
            # launcher
            {
                "target": "launcher.sh",
                "params_target": "launcher_params.yaml",
                "src": "SE_AE/launcher.sh",
                "params": {
                    "root": "/home/mdenil/code/parameter_prediction",
                    "pylearn2_data_path": "/home/mdenil/data/pylearn2",
                },
            },
            ],
        "task_params": {
            "launcher_file": "launcher.sh",
        },
    }

