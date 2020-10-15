from utilities.circuit_basics import Evaluator
import numpy as np
import matplotlib.pyplot as plt


ground = np.genfromtxt('TFIM4.csv',delimiter=',')
js,ans = ground[:,0], ground[:,1]
energies = []

for j in np.arange(0,4.0,.25):

    evaluator = Evaluator(loading=True, args={"n_qubits":4, "J":j})
    energies.append(evaluator.raw_history[len(list(evaluator.raw_history.keys()))-1][1])

plt.figure(figsize=(7,10))
ax1 = plt.subplot2grid((2,1),(0,0))
ax2 = plt.subplot2grid((2,1),(1,0))

plt.suptitle("4 qubits\n"+r'$H = -\frac{g}{2} \sum_j \sigma_j^{z} - \frac{J}{2} \sum_j \sigma_{j}^{x} \sigma_{j+1}^{x}$', size=20)
ax1.scatter(js,energies, s=20, color="red", label="ground state energy (vans)" )
ax1.plot(js, ans, color="black",label="ground state energy (numerical)")
ax2.scatter(js,np.abs((energies-ans)/ans), s=30, color="red")
ax2.set_ylabel(r'$\frac{\Delta E}{E_g}$', size=30)
ax2.set_xlabel("J", size=30)
ax1.legend()
plt.savefig("example_4.png")
#plt.plot(energies)
#plt.show()
