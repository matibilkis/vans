amplitude=$1
. ~/qenv_bilkis/bin/activate
cd ~/vans
python3 main.py --path_results "/data/uab-giq/scratch/matias/data-vans/" --qlr 0.01 --acceptange_percentage 0.1 --n_qubits 8 --reps 1000 --qepochs 1000 --problem_config '{"problem":"H4","geometry":"[('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 2)), ('H', (0.0, 0.0, 4)), ('H', (0.0, 0.0, 6))]","multiplicity":"1","charge":"0","basis":"sto-3g"}' --optimizer sgd
deactivate
