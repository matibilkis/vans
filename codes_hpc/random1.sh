#!/bin/bash
nrun=0
for nq in 2 3
do
  for ep in 1
  do
    for lr in 0.0005 0.0001 0.00005
    do
      for batch in 64 128 256
      do
        for tau in 0.01 0.1
        do
          for fits in 10 25 40
          do
            for priosc in 0.01 0.25 0.5 0.75 0.99
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
            python3 main.py --batch_size $batch --use_tqdm 0 --names "${NAME}" --total_timesteps $((nq*1000)) --n_qubits $nq --policy_agent "random" --ep $ep --episodes_before_learn 100 --depth_circuit $((2*nq)) --learning_rate $lr --tau $tau --fitsep $fits --priority_scale $priosc
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
done
