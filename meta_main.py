import os
import numpy as np


# for nq in [4]:
#     qeps = 10**4
#     for J in np.linspace(-1.1, 1.1,20):
#         instruction = "python3 main.py --J "+str(J) + " --n_qubits "+str(nq)+" --reps "+str(50)+ " --noise "+str(0)+" --qepochs "+str(qeps)+ " --g "+str(0.75) + " --problem xxz --qlr 0.01"
#         os.system(instruction)

#
for nq in [4]:
    qeps = 10**4
    for J in np.linspace(0,10,20):
        instruction = "python3 main.py --J "+str(J) + " --n_qubits "+str(nq)+" --reps "+str(50)+ " --noise "+str(0)+" --qepochs "+str(qeps)+ " --g "+str(1) + " --problem TFIM --qlr 0.01"
        os.system(instruction)
