import os
import numpy as np
import multiprocessing as mp


def dict_to_json(dictionary):
    d="{"
    for k,v in dictionary.items():
        if isinstance(k,str):
            d+='\"{}\":\"{}\",'.format(k,v)
        else:
            d+='\"{}\":{},'.format(k,v)
    d=d[:-1]
    d+="}" #kill the comma
    return "\'"+d+ "\'"

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


QUBITS = 4
QEPOCHS = 10**2
GENETIC_RUNS=10
insts=[]

problem_config = dict_to_json({"problem" : "TFIM", "g":1.0, "J": 0.3})
# problem_config = dict_to_json({"problem" : "XXZ", "g":1.0, "J": 0.3})
# BOND_LENGTH=1.2
#problem_config = dict_to_json({"problem" : "H2", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., BOND_LENGTH))], "multiplicity":1, "charge":0, "basis":"sto-3g"})

for J in [1]:
    instruction = "python3 main.py --qlr 0.01 --acceptange_percentage 0.1 --n_qubits "+str(QUBITS)+" --reps "+str(GENETIC_RUNS)+" --qepochs "+str(QEPOCHS)+ " --problem_config "+problem_config#+" --noise_config "+noise_config + "
    insts.append(instruction)

def execute_instruction(inst):
    os.system(inst)

with mp.Pool(1) as p:
    p.map(execute_instruction,insts)
