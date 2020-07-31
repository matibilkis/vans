#!/bin/bash
nrun=0
for nq in 2 3 4
do
  for ep in 0.1 0.3 1
  do
    for lr in 0.0001
    do
      for batch in 64
      do
        for tau in 0.01
        do
          for fits in 1 10 50 100
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
          #SBATCH --output=outputs/${NAME}.out\n\
          #SBATCH --error=error/${NAME}.err\n\
          #SBATCH --signal=23@60\n\
          \n\
          python3 main.py --batch_size $batch --use_tqdm 0 --names "${ep}-Greedy-${nq}-Qubi-${fits}-fits" --total_timesteps $((nq*1000)) --n_qubits $nq --policy_agent "random" --ep $ep --episodes_before_learn $((nq*100)) --depth_circuit $((2*nq)) --learning_rate $lr --tau $tau --fitsep $fits
          \n\
          echo "Stopping:"\n\
          date\n\
          "
          echo -e ${STR} | sbatch
          sleep 1m
          done
        done
      done
    done
  done
done
