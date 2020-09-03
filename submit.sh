#!/bin/bash

nl=0.05
nq=3
reps=10

fname=nq_${nq}__nl_${nl}
for j in 0.0 0.5 0.1 1.5 2 2.5
do
names=nq_${nq}__nl_${nl}__J_${j}
STR="
#!/bin/bash\n\
#SBATCH -N 1\n\
#SBATCH -t 16:00:00\n\
#SBATCH --output=outputs/${names}.out\n\
#SBATCH --error=errors/${names}.err\n\
\n\
python3 main.py --J $j --n_qubits $nq --noise_level $nl --reps $reps --names ${names} --folder_result ${fname} \n\
"
echo -e ${STR} | sbatch
done
