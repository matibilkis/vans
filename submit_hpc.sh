#!/bin/bash
for J in `seq 0 0.1 10.0`
do
STR="
#!/bin/bash\n\
#SBATCH -N 1\n\
#SBATCH -t 16:00:00\n\
#SBATCH --output=${J}.out\n\
#SBATCH --error=${J}.err\n\
\n\
python3 main.py --J $J --problem XXZ --n_qubits 8 --reps 100 \n\
"
echo -e ${STR} | sbatch
done
