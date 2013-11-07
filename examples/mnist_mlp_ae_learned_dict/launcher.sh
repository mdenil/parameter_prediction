#!/bin/bash

set -e

ROOT=/home/mdenil/bshakibi/parameter_prediction/
TRAIN_PY=$ROOT/external/pylearn2/pylearn2/scripts/train.py
export PYTHONPATH="$ROOT"
export LD_LIBRARY_PATH="$ROOT/lib:$LD_LIBRARY_PATH"
source activate "$ROOT/pp_env"

python $TRAIN_PY pretrain_layer1.yaml
python $TRAIN_PY learn_dict_layer2.yaml
python $TRAIN_PY pretrain_layer2.yaml
python $TRAIN_PY finetune_all.yaml
