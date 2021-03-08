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
# path="/data/uab-giq/scratch/matias/data-vans/"
path = "../data-vans/"
q=8
insts=[]
#st = "python3 main.py --path_results \"{}\" --qlr 0.01 --acceptance_percentage 0.001 --n_qubits {} --reps 1000 --qepochs 2000 --problem_config {} --show_tensorboarddata 0 --optimizer {} --training_patience 200 --rate_iids_per_step 1.0 --specific_name __{}__ --wait_to_get_back 25".format(path,q,problem_config, args.optimizer, args.optimizer)


# for J in np.arange(0,5.5,.5):
    # problem_config = dict_to_json({"problem" : "XXZ", "g":1.0, "J": J});q=8
# bonds = []
# bbs=[]
# for bond in np.linspace(.5,2.3,16):
#     if bond not in [.5,1.1]:
#         bbs.append(bond)
# bbs = np.linspace(.5,2.3,16)
# for bond in bbs:
# for J in np.arange(0,2,0.1):
for J in np.arange(0,2,0.1):

# for bond in []
# for init_layers, bond in enumerate([1.5]*4):
    # problem_config=dict_to_json({"problem" : "H4", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bond)), ('H', (0., 0., 2*bond)), ('H', (0., 0., 3*bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"})
    # problem_config = dict_to_json({"problem" : "H2", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"});q=4
    # problem_config = dict_to_json({"problem" : "XXZ", "g":1.0, "J": J});q=8
    problem_config = dict_to_json({"problem" : "TFIM", "g":1.0, "J": J});q=8

    instruction = "python3 main.py --path_results \"{}\" --qlr 0.01 --acceptance_percentage 0.001 --n_qubits {} --reps 100 --qepochs 10000 --problem_config {} --show_tensorboarddata 0 --optimizer adam --training_patience 1000 --rate_iids_per_step 2.0 --wait_to_get_back 10 --init_layers_hea {}".format(path,q,problem_config, 1)
    insts.append(instruction)
#
def execute_instruction(inst):
    os.system(inst)

with mp.Pool(2) as p:
    p.map(execute_instruction,insts)
