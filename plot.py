from utilities.circuit_basics import Evaluator
import numpy as np
import matplotlib.pyplot as plt


ground = np.genfromtxt('egs_TFIM.csv',delimiter=',')

print(ground)


energies = []


for j in np.arange(0,5.5,.25):

    evaluator = Evaluator(loading=True, args={"n_qubits":3, "J":j})
    energies.append(evaluator.raw_history[len(list(evaluator.raw_history.keys()))-1][-1])

print(energies)
#plt.plot(energies)
#plt.show()
