from utilities.misc import dict_to_json
import os
import argparse
#this file takes input from the submit.sh so we easily talk.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--bd", type=float, default=1.5)
args = parser.parse_args()
bd = args.bd
#/data/uab-giq/scratch/matias/data-vans/
problem_config = dict_to_json({"problem" : "H4", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bd)), ('H', (0., 0., 2*bd)), ('H', (0., 0., 3*bd))], "multiplicity":1, "charge":0, "basis":"sto-3g"});
q=8
st = "python3 main.py --path_results \"../data-vans/\" --qlr 0.01 --acceptange_percentage 0.01 --n_qubits {} --reps 10 --qepochs 1000 --problem_config {} --show_tensorboarddata 0 --optimizer sgd --training_patience 200".format(q,problem_config)
os.system(st)
