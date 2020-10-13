import os
import numpy as np
for nq in [3,4,5]:
    for J in np.arange(0, 2,1):
        # for g in np.arange(0,2,1):
        instruction = "python3 main.py --J "+str(J) + " --n_qubits "+str(nq)+" --reps 1"
        os.system(instruction)
