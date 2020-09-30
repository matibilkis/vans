#!/bin/bash

v1=0.01
v2=0.02
v3=True

name=fofo_${v1}_${v2}_${v3}

echo $name

#!/bin/bash

nl=0.05
nq=3
reps=25

for j in 0.0 0.5 0.1 1.5 2 2.5
do
name=nq_${nq}__nl_${nl}__J_${j}
echo $name
done
