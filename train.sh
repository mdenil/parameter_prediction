#!/bin/bash

HERE=$(dirname $(readlink -e $0))

export PYTHONPATH="$HERE"
source activate "$HERE/pp_env"

python external/pylearn2/pylearn2/scripts/train.py "$@"
