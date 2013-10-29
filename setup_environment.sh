#!/bin/bash

set -e

ROOT=$(dirname $(readlink -e $0))
TOTAL_PROCESSORS=$(grep processor /proc/cpuinfo | wc -l)
MAKE="make -j$TOTAL_PROCESSORS"

export LD_LIBRARY_PATH="$ROOT/lib"
export CPATH="$ROOT/include"

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

function install_gmp {
    cd "$1"
    
    wget -O- ftp://ftp.gnu.org/gnu/gmp/gmp-5.1.3.tar.bz2 | tar xj

    cd gmp-5.1.3

    ./configure --prefix="$ROOT" --enable-cxx=yes

    $MAKE
    # make check takes a long time
    #$MAKE check
    $MAKE install
}

function install_libdai {
    # requires gmp
    cd "$1"

    if [ -d "libdai" ]; then
        echo "Existing version of libdai found, removing."
        rm -rf libdai
    fi

    git clone git://git.tuebingen.mpg.de/libdai.git libdai

    cd libdai
    # HACK OMG
    git checkout 66778447e394f067b3b48d2feddb5e5578ecb03b
    cp Makefile.LINUX Makefile.conf
    # ugh...
    sed -i -e "/^CCINC/ s~$~ -I$ROOT/include~" Makefile.conf
    sed -i -e "/^CCLIB/ s~$~ -L$ROOT/lib~" Makefile.conf
    $MAKE

    cp lib/libdai.a "$ROOT/lib/."
    cp -r include/dai/ ../../include/.
}

function install_daimrf {
    # requires libdai

    EXTERNAL="$1"
    ENV="$2"

    cd "$EXTERNAL"

    if [ -d "daimrf" ]; then
        echo "Existing version of daimrf found, removing."
        rm -rf daimrf
    fi

    git clone https://github.com/amueller/daimrf.git

    cd daimrf
    # ugh...
    sed -i -e "s~g++~g++ -I$ROOT/include -L$ROOT/lib~" Makefile
    #ln -s "$ROOT/$EXTERNAL/libdai" libdai
    $MAKE

    cp daicrf.so "$ROOT/$ENV/lib/python2.7/site-packages/."
}

function install_nltk {
    conda install --yes nltk
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
safe_call install_gmp "$EXTERNAL"
safe_call install_libdai "$EXTERNAL"
safe_call install_daimrf "$EXTERNAL" "$ENV"
safe_call install_nltk

cat <<EOF

Run:

    source activate "\$(pwd)/$ENV"

to activate the environment.  When you're done you can run

    source deactivate

to close the environement.
EOF
