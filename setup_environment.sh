#!/bin/bash

set -e

ROOT=$(dirname $(readlink -e $0))
TOTAL_PROCESSORS=$(grep processor /proc/cpuinfo | wc -l)
MAKE="make -j$TOTAL_PROCESSORS"

function safe_call {
    # usage:
    #   safe_call function param1 param2 ...

    HERE=$(pwd)
    "$@"
    cd "$HERE"
}

function install_theano {
    pip install --upgrade --no-deps git+git://github.com/mdenil/Theano.git
}

function install_joblib {
    pip install joblib
}

function install_matplotlib {
    conda install --yes matplotlib
}

function install_jinja2 {
    conda install --yes jinja2
}

function install_pylearn2 {
    cd "$1"

    if [ -d "pylearn2" ]; then
        echo "Existing version of pylearn2 found, removing."
        rm -rf pylearn2
    fi

    git clone -b parameter_prediction git@bitbucket.org:mdenil/pylearn2.git 
    cd pylearn2
    python setup.py install
}

function install_libdai {
    cd "$1"

    if [ -d "libdai" ]; then
        echo "Existing version of libdai found, removing."
        rm -rf libdai
    fi

    git clone git://git.tuebingen.mpg.de/libdai.git libdai

    cd libdai
    cp Makefile.LINUX Makefile.conf
    $MAKE
}

function install_daimrf {
    # must install_libdai first
    EXTERNAL="$1"
    ENV="$2"

    cd "$EXTERNAL"

    if [ -d "daimrf" ]; then
        echo "Existing version of daimrf found, removing."
        rm -rf daimrf
    fi

    git clone https://github.com/amueller/daimrf.git

    cd daimrf
    ln -s "$ROOT/$EXTERNAL/libdai" libdai
    $MAKE

    cp daicrf.so "$ROOT/$ENV/lib/python2.7/site-packages/."
}


ENV=pp_env
EXTERNAL=external

rm -rf $ENV

conda create --yes --prefix $ENV accelerate pip nose
source activate "$(pwd)/$ENV"

safe_call install_theano
safe_call install_joblib
safe_call install_matplotlib
safe_call install_jinja2
safe_call install_pylearn2 "$EXTERNAL"
safe_call install_libdai "$EXTERNAL"
safe_call install_daimrf "$EXTERNAL" "$ENV"

cat <<EOF

Run:

    source activate "\$(pwd)/$ENV"

to activate the environment.  When you're done you can run

    source deactivate

to close the environement.
EOF
