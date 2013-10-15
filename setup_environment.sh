#!/bin/bash

function safe_call {
    # usage:
    #   safe_call function param1 param2 ...

    HERE=$(pwd)
    "$@"
    cd "$HERE"
}

function install_theano {
    pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
}

function install_joblib {
    pip install joblib
}

function install_pylearn2 {
    DIR="$1"

    cd "$DIR"
    git clone -b parameter_prediction git@bitbucket.org:mdenil/pylearn2.git 
    cd pylearn2
    python setup.py install
}

ENV=pp_env
EXTERNAL=external

rm -rf $ENV

conda create --yes --prefix pp_env accelerate pip nose
source activate "$(pwd)/$ENV"

safe_call install_theano
safe_call install_joblib
safe_call install_pylearn2 "$EXTERNAL"

cat <<EOF

Run:

    source activate "$(pwd)/$ENV"

to activate the environment.  When you're done you can run

    source deactivate

to close the environement.
EOF
