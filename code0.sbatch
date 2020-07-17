#! /bin/bash
#SBATCH --job-name=w1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=0-15:59:00
#SBATCH --no-requeue
#SBATCH --output=hpc_files/out
#SBATCH --error=hpc_files/err
#SBATCH --qos=standard
# #SBATCH --ntasks=1
# #SBATCH -p shared
# #SBATCH --nodelist=cn452
set -x                          # Output commands
set -e                          # Abort on errors
echo "Checking:"
pwd
hostname
date
env | sort > ENVIRONMENT
echo "Starting: code vans"

python3 main_dueldqn.py --n_qubits 3 --total_timesteps 10000 --episodes_before_learn 1000 --depth_circuit 3

# put your command here!
echo "Stopping:"
date
echo "Done! :)"
