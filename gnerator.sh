#!/bin/bash
nrun=0
for nq in 2
do
  for ep in 1 0.1 0.01
  do
    for lr in 0.01 0.001 0.0001
    do
      for batch in 8 32 64
      do
        for tau in 0.01 0.1
        do
          for priosc in 0.001 0.01 0.25 0.5 0.75
          do
            NAME=run_${nrun}
            nrun=$(($nrun +1))
            nrun1=0
            for reps in 25
            do
            NAME1=run_${nrun1}
            nrun1=$(($nrun1 +1))
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
            python3 dqn_train_from_dicts.py --batch_size $batch --names "${NAME}_${NAME1}" --ep $ep --lr $lr --tau $tau --priority_scale $priosc
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
    done
  done
done
