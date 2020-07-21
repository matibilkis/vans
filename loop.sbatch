#!/bin/bash
nrun=0
for lr in 0.001 0.0001
do
  for tau in 0.01 0.001
  do

    for priosc in 0.2 0.5
    do
    NAME=run_${nrun}
    nrun=$(($nrun +1))

    STR="
    #! /bin/bash
    #SBATCH --job-name=${NAME}\n\
    #SBATCH --nodes=1\n\
    #SBATCH --exclusive\n\
    #SBATCH --mail-user=bilkis@lanl.gov\n\
    #SBATCH --mail-type=END\n\
    #SBATCH --time=0-15:59:00\n\
    #SBATCH --no-requeue\n\
    #SBATCH --output=output/${nrun}.out\n\
    #SBATCH --error=error/${nrun}.err\n\
    #SBATCH --qos=standard\n\
    #SBATCH --mem 100000\n\
    #SBATCH --nodes=1\n\
    ​
    set -x\n\                          # Output commands
    set -e\n\                          # Abort on errors
    pwd\n\
    hostname\n\
    date\n\
    env | sort > ENVIRONMENT\n\
    echo "Starting"\n\


    python3 main_dueldqn.py --learning_rate $lr --tau $tau --priority_scale $priosc\n\
    "

    echo -e ${STR} | sbatch
    sleep 1s
    done
  done
done
