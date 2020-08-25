#!/bin/bash

for v in 0.01
do
STR="
#!/bin/bash\n\
#SBATCH -N 1\n\
#SBATCH -t 16:00:00\n\
#SBATCH --output=${v}.out\n\
#SBATCH --error=${v}.err\n\
\n\
python3 main.py --J $v \n\
"
echo -e ${STR} | sbatch
done
