#!/bin/bash

nl=0.05
nq=3
reps=10

for j in 0.0 0.5 0.1 1.5 2 2.5
do
names=nq_${nq}__nl_${nl}__J_${j}
STR="
main.py --J $j --n_qubits $nq --noise_level $nl --reps $reps --names ${names} \n\
"
echo python3 ${STR} | python3

done
