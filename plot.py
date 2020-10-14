from utilities.circuit_basics import Evaluator
import numpy as np
import matplotlib.pyplot as plt
energies = []
for j in np.arange(0,6,.25):

    evaluator = Evaluator(loading=True, args={"n_qubits":3, "J":j})
    unitary, energy = evaluator.raw_history[len(list(evaluator.keys()))]
    energies.append(energy)

plt.plot(energies)
plt.show()
