#!/bin/bash
for vvv in 0.01
do

  STR="
  #!/bin/bash\n\
  #SBATCH -N 1\n\
  #SBATCH -t 2:00\n\
  \n\
  python3 print.py
  \n\
  "
  echo -e ${STR} | sbatch
done



#!/bin/bash

STR="
#!/bin/bash\n\
#SBATCH -N 1\n\
#SBATCH -t 2:00\n\
\n\
python3 print.py
\n\
"
echo -e ${STR} | sbatch



  STR="
  #!/bin/bash
  #SBATCH --N 1\n\
  #SBATCH --t 2:00\n\
  #SBATCH --output=${value}.out\n\
  #SBATCH --error=${value}.err\n\
  \n\
  python3 main.py --J $value
  \n\
  echo "Stopping:"\n\
  date\n\
  "
  echo -e ${STR} | sbatch
  sleep 1s
done
