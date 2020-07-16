#! /bin/bash
#SBATCH --job-name=w1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=0-3:00:00
#SBATCH --no-requeue
#SBATCH --output=out
#SBATCH --error=err
#SBATCH --qos=standard
# #SBATCH --ntasks=40
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
python3 main_dueldqn.py
# put your command here!
echo "Stopping:"
date
echo "Done! :)"
