#!/bin/bash
for argu in `seq 0.3 .225 1.875`
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
python3 simulbash.py --bond $argu\n\
"
echo -e ${STR} | sbatch
done
