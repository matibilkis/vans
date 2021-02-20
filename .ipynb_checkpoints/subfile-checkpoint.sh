#!/bin/bash
### this file should be modified accordingly to the hyperparameters that one desires to run
swvar1=$1
. ~/qenv_bilkis/bin/activate
cd ~/vans
python3 simulate_bash.py --swvar1 $swvar1
deactivate
