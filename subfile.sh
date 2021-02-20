#!/bin/bash
### this file should be modified accordingly to the hyperparameters that one desires to run
rates=$1
. ~/qenv_bilkis/bin/activate
cd ~/vans
python3 simulate_bash.py --ratesiid $rates
deactivate
