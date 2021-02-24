from utilities.misc import dict_to_json
import os
import argparse
#this file takes input from the submit.sh so we easily talk.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--bonds", type=float, default=1.5)
parser.add_argument("--ratesiid", type=float, default=1.)
parser.add_argument("--nrun", type=float, default=1.)

args = parser.parse_args()
bond=args.bonds
ratesiid=args.ratesiid
nrun=args.nrun
problem_config = dict_to_json({"problem" : "H4", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bond)), ('H', (0., 0., 2*bond)), ('H', (0., 0., 3*bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"});q=8
############## CONDENSED MATTER ############
            ##### XXZ #######
#problem_config = dict_to_json({"problem" : "XXZ", "g":1.0, "J": args.swvar1});q=2
            ##### tfim #######
#problem_config = dict_to_json({"problem" : "tfim", "g":1.0, "J": 0.3})


### POSSIBLE PATHS
# path="/data/uab-giq/scratch/matias/data-vans/"
path = "../data-vans/"
st = "python3 main.py --path_results \"{}\" --qlr 0.01 --acceptance_percentage 0.01 --n_qubits {} --reps 3000 --qepochs 2000 --problem_config {} --show_tensorboarddata 0 --optimizer adam --training_patience 200 --rate_iids_per_step {}".format(path,q,problem_config, ratesiid)
os.system(st)
