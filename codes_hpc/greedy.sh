#!/bin/bash
nrun=0
for lr in 0.1 0.01
do
  for tau in 0.01 0.001
  do
    for priosc in 0.1 0.7 0.9
    do
    NAME=run_${nrun}
    nrun=$(($nrun +1))

    STR="
    #!/bin/tcsh\n\
    #SBATCH --time=15:59:00\n\
    #SBATCH --nodes=1\n\
    #SBATCH --ntasks=1\n\
    #SBATCH --job-name=w1\n\
    #SBATCH --qos=standard\n\
    #SBATCH --mail-user=bilkis@lanl.gov\n\
    #SBATCH --mail-type=END\n\
    #SBATCH --no-requeue\n\
    #SBATCH --output==outputs/${NAME}\n\
    #SBATCH --error=errs/${NAME}\n\
    #SBATCH --signal=23@60\n\
    \n\
    python3 main_dueldqn.py --names "01GreedyPolicy" --n_qubits 3 --policy_agent "random" --ep 0.1 --total_timesteps 2000 --episodes_before_learn 100 --depth_circuit 3 --learning_rate $lr --tau $tau --priority_scale $priosc
    \n\
    echo "Stopping:"\n\
    date\n\
    "
    echo -e ${STR} | sbatch
    done
  done
done
