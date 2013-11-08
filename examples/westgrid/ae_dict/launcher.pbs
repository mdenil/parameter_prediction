#!/bin/bash
#PBS -S /bin/bash
#PBS -q gpu
#PBS -l nodes=1:ppn=12:gpus=1
#PBS -l walltime=10:00:00
#PBS -l mem=10gb
#PBS -j oe

cd $PBS_O_WORKDIR
echo "Current working directory is `pwd`"

echo "Node file: $PBS_NODEFILE :"
echo "---------------------"
cat $PBS_NODEFILE
echo "---------------------"
PBS_NP=`/bin/awk 'END {print NR}' $PBS_NODEFILE`
echo "Running on $PBS_NP processors."

echo "GPU file: $PBS_GPUFILE :"
echo "------------------"
cat $PBS_GPUFILE
GPU_FULLNAME=`cat $PBS_GPUFILE`
echo "------------------"
NUM_GPUS=`/bin/awk 'END {print NR}' $PBS_GPUFILE`
echo "$NUM_GPUS GPUs assigned."

echo "Starting at `date`"
nvidia-debugdump -l
nvidia-smi
GPU_NAME=`echo $GPU_FULLNAME|cut -d'-' -f 2`
echo $GPU_NAME

set -e
ROOT=/home/bshakibi/pp/parameter_prediction/
TRAIN_PY=$ROOT/external/pylearn2/pylearn2/scripts/train.py
export PYTHONPATH="$ROOT"
export LD_LIBRARY_PATH="$ROOT/lib:$LD_LIBRARY_PATH"

source activate "$ROOT/pp_env"

THEANO_COMPILEDIR=/tmp/$PBS_JOBID
mkdir -p $THEANO_COMPILEDIR

module load cuda

THEANO_FLAGS=mode=FAST_RUN,device=$GPU_NAME,floatX=float32,base_compiledir=$THEANO_COMPILEDIR python $TRAIN_PY learn_dict_layer1.yaml
THEANO_FLAGS=mode=FAST_RUN,device=$GPU_NAME,floatX=float32,base_compiledir=$THEANO_COMPILEDIR python $TRAIN_PY pretrain_layer1.yaml

THEANO_FLAGS=mode=FAST_RUN,device=$GPU_NAME,floatX=float32,base_compiledir=$THEANO_COMPILEDIR python $TRAIN_PY learn_dict_layer2.yaml
THEANO_FLAGS=mode=FAST_RUN,device=$GPU_NAME,floatX=float32,base_compiledir=$THEANO_COMPILEDIR python $TRAIN_PY pretrain_layer2.yaml

THEANO_FLAGS=mode=FAST_RUN,device=$GPU_NAME,floatX=float32,base_compiledir=$THEANO_COMPILEDIR python $TRAIN_PY finetune_all.yaml


echo "Finished at `date`"
