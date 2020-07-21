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
    #! /bin/bash\n\
    #SBATCH --job-name=w1\n\
    #SBATCH --nodes=1\n\
    #SBATCH --exclusive\n\
    #SBATCH --time=0-15:59:00\n\
    #SBATCH --no-requeue\n\
    #SBATCH --output=${nrun}.out\n\
    #SBATCH --error=${nrun}.err\n\
    #SBATCH --qos=standard\n\
    # #SBATCH --ntasks=1\n\
    # #SBATCH -p shared\n\
    # #SBATCH --nodelist=cn452\n\
    ​
    set -x\n\                          # Output commands
    set -e\n\                          # Abort on errors
    pwd\n\
    hostname\n\
    date\n\
    env | sort > ENVIRONMENT\n\
    echo "Starting"\n\


    python3 script_test_segmentation.py\n\
    "

    echo -e ${STR} | sbatch
    sleep 1s
    done
  done
done
