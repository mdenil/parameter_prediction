#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 tool_script.py arg0 arg1 ..."
    exit 1;
fi

HERE=$(dirname $(readlink -e $0))
TOOLPATH="$HERE/tools:$HERE/external/pylearn2/pylearn2/scripts"
export PYTHONPATH="$HERE"
export LD_LIBRARY_PATH="$HERE/lib:$LD_LIBRARY_PATH"
source activate "$HERE/pp_env"

TOOL_NAME="$1";
TOOL_PARAMS="${@:2}"

for D in ${TOOLPATH//:/ }; do
    if [ -f "$D/$TOOL_NAME" ]; then
        TOOL="$D/$TOOL_NAME"
        break
    fi
done

echo "Running: '$TOOL'"
echo "Params : $TOOL_PARAMS"

python "$TOOL" $TOOL_PARAMS
