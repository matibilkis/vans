import os
import numpy as np
import multiprocessing as mp


def channel_dict(channel="depolarizing", channel_params=[0], q_batch_size=10**2):
    d= '{\"channel\":\"' +channel + '\",\"channel_params\":'+str(channel_params)+',\"q_batch_size\":' + str(q_batch_size) + '}'
    return "\'"+d+ "\'"


nq = 3
qeps = 10**4
J=np.linspace(0,10,20)[1]
# instruction = "python3 main.py --J "+str(J) + " --n_qubits "+str(nq)+" --reps "+str(100)+" --qepochs "+str(qeps)+ " --g "+str(1) + " --problem TFIM --qlr 0.005"
# os.system(instruction)

for p in [10**-6, 10**-5, 10**-4, 10**-3, 10**-2]:
    noise_model = channel_dict(channel_params=[p], q_batch_size=10**3)
    instruction = "python3 main.py --J "+str(J) + " --n_qubits "+str(nq)+" --reps "+str(50)+" --qepochs "+str(qeps)+ " --g "+str(1) + " --problem TFIM --qlr 0.005" +" --noise_model "+noise_model + " --verbose 0"
    os.system(instruction)