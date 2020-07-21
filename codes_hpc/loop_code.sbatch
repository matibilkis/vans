#!/bin/bash
nrun=0
for lr in 0.01
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

python3 main_dueldqn.py --n_qubits 2 --total_timesteps 10 --episodes_before_learn 2 --depth_circuit 4 --use_tqdm 1 --learning_rate $lr --tau $tau \n\
"

echo -e ${STR} | sbatch
sleep 1s
done
