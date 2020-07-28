#!/bin/bash
nrun=0
for ep in 1 0.75 0.5 0.3 0.1 0.01
do
  for lr in 0.000001 0.00001 0.0001
  do
    for batch in 64 128 256
    do
      for tau in 0.01 0.001
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
      python3 working_out.py --batch_size $batch --use_tqdm 1 --names "WorkingOut" --total_timesteps 1000 --n_qubits 2 --policy_agent "random" --ep $ep --episodes_before_learn 100 --depth_circuit 3 --learning_rate $lr --tau $tau
      \n\
      echo "Stopping:"\n\
      date\n\
      "
      echo -e ${STR} | sbatch
      sleep 1s
      done
    done
  done
done
