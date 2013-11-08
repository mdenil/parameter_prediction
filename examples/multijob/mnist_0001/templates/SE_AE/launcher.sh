#!/bin/bash

#!/bin/bash
#PBS -S /bin/bash

# Which queue
##PBS -l qos=debug

# job array indexes
##PBS -t 0-49
# job settings
#PBS -l procs=1
#PBS -l walltime=15:00:00
#PBS -l mem=2gb
# restartable?
#PBS -r n
# merege stderr into stdout
#PBS -j oe
##PBS -e $PBS_JOBID.err
#PBS -o {{job_dir}}/$PBS_JOBID.out
# job name
#PBS -N SE_AE_{{job_id}}
# email on job completion
#PBS -M misha.denil@gmail.com
#PBS -m bea
# pass environment variables to the job
##PBS -V

# http://www.westgrid.ca/support/running_jobs

set -e

export MKL_NUM_THREADS=1
export PYLEARN2_DATA_PATH="{{pylearn2_data_path}}"

TRAIN_PY="{{root}}/external/pylearn2/pylearn2/scripts/train.py"
export PYTHONPATH="{{root}}"
export LD_LIBRARY_PATH="{{root}}/lib:$LD_LIBRARY_PATH"
source activate "{{root}}/pp_env"

echo -n "BEGIN: "; date '+%s'

python $TRAIN_PY {{job_dir}}/pretrain_layer1.yaml
python $TRAIN_PY {{job_dir}}/learn_dict_layer2.yaml
python $TRAIN_PY {{job_dir}}/pretrain_layer2.yaml
python $TRAIN_PY {{job_dir}}/finetune_all.yaml

echo -n "END: "; date '+%s'

