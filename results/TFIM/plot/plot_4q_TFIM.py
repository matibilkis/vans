import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utilities.circuit_basics import Evaluator


# matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
# matplotlib.rc('font', serif='cm10')
# matplotlib.rc('text', usetex=True)
# #
axinticks=[]
plt.rcParams.update({'font.size': 45})


ground = np.genfromtxt('TFIM/plot/TFIM4_v2.csv',delimiter=',')
js,ans = ground[:,0], ground[:,1]
energies = []

for j in np.linspace(0,10,20):

    evaluator = Evaluator(loading=True, args={"n_qubits":4, "J":j})
    energies.append(evaluator.raw_history[len(list(evaluator.raw_history.keys()))-1][-1])

plt.figure(figsize=(10,10))
ax1 = plt.subplot2grid((2,1),(0,0))
ax2 = plt.subplot2grid((2,1),(1,0))

plt.suptitle("4 qubits\n"+r'$H = -g \sum_j \sigma_j^{z} - J \sum_j \sigma_{j}^{x} \sigma_{j+1}^{x}$', size=20)
ax1.scatter(js,energies, s=20, color="red", label="lowest energy found (vans)" )
ax1.plot(js, ans, color="black",label="true ground state energy")
ax2.scatter(js,np.abs((energies-ans)/ans), s=30, color="red")
ax2.set_ylabel(r'$\frac{\Delta E}{E_g}$', size=20)
ax2.set_xlabel("J", size=30)
ax1.legend(prop={"size":15})
# plt.show()
plt.savefig("TFIM/tfim4.png")
# plt.plot(energies)
# plt.show()
