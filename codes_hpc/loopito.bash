#!/bin/bash
for LR in 0.01 0.001
do
python3 main_dueldqn.py --n_qubits 2 --total_timesteps 10 --episodes_before_learn 2 --depth_circuit 4 --use_tqdm 1 --learning_rate $LR
done
