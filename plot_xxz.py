from utilities.circuit_basics import Evaluator
import numpy as np
import matplotlib.pyplot as plt


ground = np.genfromtxt('xxz1.csv',delimiter=',')
js1,ans = ground[:,0], ground[:,1]
energies = []
# js = np.linspace(-1.1,1.1,20)
js = np.linspace(0,10,4)
for j in js:
    evaluator = Evaluator(loading=True, args={"n_qubits":4, "J":j, "g":1.,"problem":"xxz"})
    energies.append(evaluator.raw_history[len(list(evaluator.raw_history.keys()))-1][1])

plt.figure(figsize=(10,10))
ax1 = plt.subplot2grid((2,1),(0,0))
ax2 = plt.subplot2grid((2,1),(1,0))

plt.suptitle("4 qubits\n"+r'$H = g \sum_j \sigma_j^{z} + \sum_j \sigma_{j}^{x} \sigma_{j+1}^{x} +  \sigma_{j}^{y} \sigma_{j+1}^{y} + J  \sigma_{j}^{z} \sigma_{j+1}^{z}$', size=20)
# ax1.scatter(js,energies, s=20, color="red", label="ground state energy (vans)" )
ax1.plot(js1, ans, color="black",label="ground state energy (numerical)")
# ax2.scatter(js,np.abs((energies-ans)/ans), s=30, color="red")
ax2.set_ylabel(r'$\frac{\Delta E}{E_g}$', size=30)
ax2.set_xlabel("J", size=30)
ax1.legend()
# plt.savefig("xxz_4q_20_10.png")
#plt.plot(energies)
plt.show()
