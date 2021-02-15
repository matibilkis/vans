import os
import numpy as np
import multiprocessing as mp


def dict_to_json(dictionary):
    d="{"
    for k,v in dictionary.items():
        if type(k) == 'str':
            d+='\"{}\":\"{}\",'.format(k,v)
        else:
            d+='\"{}\":{},'.format(k,v)
    d=d[:-1]
    d+="}" #kill the comma
    return "\'"+d+ "\'"

##########################################################################################################################################################################
##########################################################################################################################################################################
##########                  models for dicts               ###############################################################################################################
##########  noise_model = {"channel": "depolarizing", "channel_params"=[0], "q_batch_size":10**2}
##########  problem_config = {"problem" : "H2", geometry: [('H', (0., 0., 0.)), ('H', (0., 0., BOND_LENGTH))]
##########  problem_config = {"problem" : "XXZ", "g":1.0, "J": 0.3]
##########  problem_config = {"problem" : "TFIM", "g":1.0, "J": 0.3]
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################


def channel_dict(channel="depolarizing", channel_params=[0], q_batch_size=10**2):
    d= '{\"channel\":\"' +channel + '\",\"channel_params\":'+str(channel_params)+',\"q_batch_size\":' + str(q_batch_size) + '}'
    return "\'"+d+ "\'"






nq = 8
qeps = 10**4
genetic_runs=250
insts=[]
# for p in [10**-7, 10**-6, 10**-5, 10**-4]:
# for p in [10**-8, 10**-3, 10**-2, 10**-1]:
for J in np.linspace(0,10,4):
    # noise_model = channel_dict(channel_params=[p], q_batch_size=10**3)
    instruction = "python3 main.py --J "+str(J) + " --n_qubits "+str(nq)+" --reps "+str(genetic_runs)+" --qepochs "+str(qeps)+ " --g "+str(1) + " --problem XXZ --qlr 0.005" #+" --noise_model "+noise_model + " --verbose 0"
    # os.system(instruction)
    insts.append(instruction)

def execute_instruction(inst):
    os.system(inst)

with mp.Pool(1) as p:
    p.map(execute_instruction,insts)
