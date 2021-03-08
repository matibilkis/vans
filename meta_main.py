import os
import numpy as np
import multiprocessing as mp
from utilities.misc import dict_to_json

# if problem == "TFIM":
#     problem_config = dict_to_json({"problem" : "TFIM", "g":1.0, "J": args.J});q=4
# elif problem == "XXZ":
#     problem_config = dict_to_json({"problem" : "XXZ", "g":1.0, "J": args.J});q=4

################### molecular ###############
# else:
#     problem_config = dict_to_json({"problem" : "H4", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bond)), ('H', (0., 0., 2*bond)), ('H', (0., 0., 3*bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"});q=8
# for bond in [1.5]*4:
    # problem_config = dict_to_json({"problem" : "XXZ", "g":1.0, "J": J});q=8


### POSSIBLE PATHS
path="/data/uab-giq/scratch/matias/data-vans/"
# path = "../data-vans/"



#st = "python3 main.py --path_results \"{}\" --qlr 0.01 --acceptance_percentage 0.001 --n_qubits {} --reps 1000 --qepochs 2000 --problem_config {} --show_tensorboarddata 0 --optimizer {} --training_patience 200 --rate_iids_per_step 1.0 --specific_name __{}__ --wait_to_get_back 25".format(path,q,problem_config, args.optimizer, args.optimizer)
# for init_layers, bond in enumerate([1.5]*4):
    # problem_config=dict_to_json({"problem" : "H4", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bond)), ('H', (0., 0., 2*bond)), ('H', (0., 0., 3*bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"})
    # problem_config = dict_to_json({"problem" : "H2", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"});q=4
    # problem_config = dict_to_json({"problem" : "XXZ", "g":1.0, "J": J});q=8

js = np.arange(-4,4.2,.2)
for J in js:

    problem_config = dict_to_json({"problem" : "XXZ", "g":1.0, "J": J});q=8

    instruction = "python3 main.py --path_results \"{}\" --qlr 0.01 --acceptance_percentage 0.01 --n_qubits {} --reps 200 --qepochs 10000 --problem_config {} --optimizer adam --training_patience 1000 --rate_iids_per_step 3.0 --wait_to_get_back 15 --init_layers_hea {} --reduce_acceptance_percentage 0".format(path,q,problem_config, 3)
    insts.append(instruction)
#
def execute_instruction(inst):
    os.system(inst)

with mp.Pool(4) as p:
    p.map(execute_instruction,insts)
