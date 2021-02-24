#!/bin/bash
for argu in `seq -2.0 .1 2.`
do
STR="
#!/bin/bash\n\
#SBATCH -N 1\n\
#SBATCH --exclusive=user\n\
#SBATCH --job-name=qvans\n\
#SBATCH --time=2-0:00:00\n\
#SBATCH --output=${argu}.out\n\
#SBATCH --error=${argu}.err\n\
#SBATCH --qos=long\n\
. qvans/bin/activate\n\
python3 simulbash.py --J $J\n\
"
echo -e ${STR} | sbatch
done
