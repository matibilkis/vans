import os
import numpy as np

for noise in [0, 0.01]:
    for nq in [4]:
        if noise>0:
            qeps = 500*nq
        else:
            qeps = 2000*nq
        for J in np.arange(0, 4,.25):
            instruction = "python3 main.py --J "+str(J) + " --n_qubits "+str(nq)+" --reps "+str(20*nq)+ " --noise "+str(noise)+" --qepochs "+str(qeps)+ " --g "+str(1)
            os.system(instruction)
