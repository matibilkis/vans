import os
import numpy as np
import multiprocessing as mp
from utilities.misc import dict_to_json

##########################################################################################################################################################################
##########################################################################################################################################################################
##########                  models for dicts               ###############################################################################################################
##########  noise_config = {"shots":1000, "channel": "depolarizing", "channel_params"=[0], "q_batch_size":10**2}
##########  problem_config = {"problem" : "H2", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., BOND_LENGTH)), "multiplicity":1, "charge":0, "basis":"sto-3g"}
##########  problem_config = {"problem" : "XXZ", "g":1.0, "J": 0.3}
##########  problem_config = {"problem" : "TFIM", "g":1.0, "J": 0.3}
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################



QEPOCHS = 10**3
GENETIC_RUNS=5
insts=[]
# problem_config = dict_to_json({"problem" : "TFIM", "g":1.0, "J": 0.3})
# problem_config = dict_to_json({"problem" : "XXZ", "g":1.0, "J": 0.3})
# BOND_LENGTH=1.2
problem_config = dict_to_json({"problem" : "H4", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., 1.5)), ('H', (0., 0., 3.0)), ('H', (0., 0., 4.5))], "multiplicity":1, "charge":0, "basis":"sto-3g"});QUBITS = 8
# noise_config = dict_to_json({"channel": "depolarizing", "channel_params":[0], "q_batch_size":10**2})
#problem_config = dict_to_json({"problem" : "LiH", "geometry": [('Li', (0., 0., 0.)), ('H', (0., 0., 2.0))], "multiplicity":1, "charge":0, "basis":"sto-3g"}); QUBITS=12
# problem_config = dict_to_json({"problem" : "H4", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., 1.5)), ('H', (0., 0., 3.0)), ('H', (0., 0., 4.5))], "multiplicity":1, "charge":0, "basis":"sto-3g"}); QUBITS=8

for J in [1]:
    instruction = "python3 main.py --path_results "+"\"../data-vans/\""+" --qlr 0.01 --acceptange_percentage 0.01 --n_qubits "+str(QUBITS)+" --reps "+str(GENETIC_RUNS)+" --qepochs "+str(QEPOCHS)+ " --problem_config "+problem_config#+" --noise_config "+noise_config + "
    insts.append(instruction)
#
print(instruction)

os.system(instruction)
def execute_instruction(inst):
    os.system(inst)

with mp.Pool(1) as p:
    p.map(execute_instruction,insts)
