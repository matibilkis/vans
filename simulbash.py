from utilities.misc import dict_to_json
import os
import argparse
import numpy as np
#this file takes input from the submit.sh so we easily talk.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--bonds", type=float, default=1.5)
parser.add_argument("--ratesiid", type=float, default=1.)
parser.add_argument("--nrun", type=float, default=1.)
parser.add_argument("--J", type=float, default=2.)
parser.add_argument("--optimizer", type=str, default="adam")

args = parser.parse_args()
bond=np.round(args.bonds,2)
ratesiid=args.ratesiid
nrun=args.nrun

#problem_config = dict_to_json({"problem" : "H4", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bond)), ('H', (0., 0., 2*bond)), ('H', (0., 0., 3*bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"});q=8

#problem_config = dict_to_json({"problem" : "LiH", "geometry": [('Li', (0., 0., 0.)), ('H', (0., 0., bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"});q=12

############## CONDENSED MATTER ############
            ##### XXZ #######
#problem_config = dict_to_json({"problem" : "XXZ", "g":1.0, "J": args.J});q=12
            ##### tfim #######
problem_config = dict_to_json({"problem" : "TFIM", "g":1.0, "J": 3.2});q=4
### POSSIBLE PATHS
# path="/data/uab-giq/scratch/matias/data-vans/"
path = "../data-vans/"
#st = "python3 main.py --path_results \"{}\" --qlr 0.01 --acceptance_percentage 0.001 --n_qubits {} --reps 1000 --qepochs 2000 --problem_config {} --show_tensorboarddata 0 --optimizer {} --training_patience 200 --rate_iids_per_step 1.0 --specific_name __{}__ --wait_to_get_back 25".format(path,q,problem_config, args.optimizer, args.optimizer)

st = "python3 main.py --path_results \"{}\" --qlr 0.01 --acceptance_percentage 0.001 --n_qubits {} --reps 1000 --qepochs 2000 --problem_config {} --show_tensorboarddata 0 --optimizer {} --training_patience 200 --rate_iids_per_step 1.0 --wait_to_get_back 25".format(path,q,problem_config, args.optimizer, args.optimizer)

os.system(st)
