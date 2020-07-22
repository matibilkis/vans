#!/bin/tcsh

#Submit this script with: sbatch filename

#SBATCH --time=15:59:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --job-name=w1   # job name
#SBATCH --qos=standard   # qos name
#SBATCH --mail-user=bilkis@lanl.gov   # email address
#SBATCH --mail-type=END
#SBATCH --no-requeue   # do not requeue when preempted and on node failure
#SBATCH --signal=23@60  # send signal to job at [seconds] before end

python3 main_dueldqn.py --n_qubits 3 --total_timesteps 1000 --episodes_before_learn 100 --depth_circuit 3 --learning_rate 0.001 --tau 0.05

# put your command here!
echo "Stopping:"
date
