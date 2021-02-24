#!/bin/bash
### this file should be modified accordingly to the hyperparameters that one desires to run
iid=$1
optimizer=$2
cd ~/vans
. ~/qenv_bilkis/bin/activate
#python3 simulate_bash.py --ratesiid $rates
python3 simulbash.py --bonds $iid --optimizer $optimizer
deactivate
