#!/bin/bash
set -e

export MKL_NUM_THREADS=1
export PYLEARN2_DATA_PATH="/home/mdenil/data/pylearn2"

TRAIN_PY="/home/mdenil/code/parameter_prediction/external/pylearn2/pylearn2/scripts/train.py"
export PYTHONPATH="/home/mdenil/code/parameter_prediction"
export LD_LIBRARY_PATH="/home/mdenil/code/parameter_prediction/lib:$LD_LIBRARY_PATH"
source activate "/home/mdenil/code/parameter_prediction/pp_env"

echo -n "BEGIN: "; date '+%s'

python $TRAIN_PY pretrain_layer1.yaml
python $TRAIN_PY pretrain_layer2.yaml
python $TRAIN_PY finetune_all.yaml

echo -n "END: "; date '+%s'
