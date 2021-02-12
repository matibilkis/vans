import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
from tqdm import tqdm
sys.path[0] = "/home/cooper-cooper/Desktop/vans/"

from utilities.circuit_basics import Evaluator
import os

matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
matplotlib.rc('font', serif='cm10')
matplotlib.rc('text', usetex=True)
# #
axinticks=[]
plt.rcParams.update({'font.size': 45})


ground = np.genfromtxt('results/TFIM/plot/TFIM8.csv',delimiter=',')
js,ans = ground[:,0], ground[:,1]
energies = []
#
for j in tqdm(np.arange(0,4.7,0.1)):

    evaluator = Evaluator(loading=True, args={"n_qubits":8, "J":np.round(j,2)})
    energies.append(evaluator.raw_history[len(list(evaluator.raw_history.keys()))-1][-1])
    np.save("results/TFIM/plot/energiesTFIM8_lowest",energies)
#
energies = np.load("results/TFIM/plot/energiesTFIM8.npy")
plt.figure(figsize=(20,20))
ax1 = plt.subplot2grid((2,1),(0,0))
ax2 = plt.subplot2grid((2,1),(1,0))

plt.suptitle("8 qubits\n"+r'$H = -g \sum_j \sigma_j^{z} - J \sum_j \sigma_{j}^{x} \sigma_{j+1}^{x}$', size=20)
ax1.scatter(js,energies, s=20, color="red", label="lowest energy found (vans)" )
ax1.plot(js, ans, color="black",label="true ground state energy")
ax2.scatter(js,np.abs((energies-ans)/ans), s=30, color="red")
ax2.set_ylabel(r'$\frac{\Delta E}{E_g}$', size=20)
ax2.set_xlabel("J", size=30)
#for a in [ax1, ax2]:#
    #a.set_xticks(np.arange(0,4.6,0.4))
ax1.legend(prop={"size":15})
# plt.show()
plt.savefig("TFIM/tfim8.png")
# # plt.plot(energies)
# # plt.show()
