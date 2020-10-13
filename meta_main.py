import os
import numpy as np

for noise in [0, 0.01, 0.05, 0.1]:
    for nq in [3,4,5]:
        if noise>0:
            qeps = 500*nq
        else:
            qeps = 2000*nq
        for J in np.arange(.1, 1,.1):
            for g in np.arange(0,1,.1):
                if J == .1 and g< 0.31:
                    pass
                else:

                    instruction = "python3 main.py --J "+str(J) + " --n_qubits "+str(nq)+" --reps "+str(10*nq)+ " --noise "+str(noise)+" --qepochs "+str(qeps)+ " --g "+str(g)
                    os.system(instruction)
