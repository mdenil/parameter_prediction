import numpy as np
import os
import sys

data_dir = "/global/scratch/bshakibi/data/timit"
base_dir = "scratch_space_COV"
template_dir = "templates/COV"
NVIS = 429

# This dictionary is available to all templates.
# You can add parameters here.
global_template_params = {
    }

def _n_atoms_per_column(n_vis, n_columns, prop):
    return int(int(n_vis / n_columns) * prop)

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
                "src": "pretrain_layer1.yaml",
                "params": {
                    "n_epochs": n_epochs[0],
                    "n_vis": NVIS,
                    "columns": [
                        {
                            "n_atoms": _n_atoms_per_column(NVIS, n_columns, prop),
                            "n_hid": _n_hid_per_column(n_hid, n_columns)
                        },
                    ] * n_columns,
                },
            },

            {
                "target": "pretrain_layer2.yaml",
                "params_target": "pretrain_layer2_params.yaml",
                "src": "pretrain_layer2.yaml",
                "params": {
                    "n_epochs": n_epochs[1],
                    "n_vis": _n_hid_total(n_hid, n_columns),
                    "columns": [
                        {
                            "n_atoms": _n_atoms_per_column(_n_hid_total(n_hid, n_columns), n_columns, prop),
                            "n_hid": _n_hid_per_column(n_hid, n_columns)
                        },
                    ] * n_columns,
                },
            },
            
            {
                "target": "finetune_all.yaml",
                "params_target": "finetune_all_params.yaml",
                "src": "finetune_all.yaml",
                "params": {
                    "n_epochs": n_epochs[2],
                    "n_vis": NVIS,
                },
            },
            
            # launcher
            {
                "target": "launcher.sh",
                "params_target": "launcher_params.yaml",
                "src": "launcher.pbs",
                "params": {
                    "data_dir": data_dir,
                },
            },
            ],
        "task_params": {
            "launcher_file": "launcher.sh",
        },
    }

