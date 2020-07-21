#!/bin/bash
nrun=0
for lr in 0.01 0.05
do
  for tau in 0.01 0.05
  do

  NAME=run_${nrun}
  nrun=$(($nrun +1))


  STR="
  #! /bin/bash

  python3 main_dueldqn.py --n_qubits 2 --total_timesteps 10 --episodes_before_learn 2 --depth_circuit 4 --use_tqdm 1 --learning_rate $lr --tau $tau\n\
  "

  echo -e ${STR}
  done
done
