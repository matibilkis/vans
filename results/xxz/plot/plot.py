import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
from tqdm import tqdm
sys.path[0] = "/home/cooper-cooper/Desktop/vans/"
from utilities.evaluator import Evaluator
import os

print(os.getcwd())

matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
matplotlib.rc('font', serif='cm10')
matplotlib.rc('text', usetex=True)
# #
axinticks=[]
plt.rcParams.update({'font.size': 45})

#
ground = np.genfromtxt('results/xxz/plot/xxz14.csv',delimiter=',')
js,ans = ground[:,0], ground[:,1]
energies = []
# #
# for j in tqdm(js):
#     args={"n_qubits":8,"problem_config":{"problem" : "XXZ", "g":1.0, "J": j}, "load_displaying":False,"specific_folder_name":"14Q - J {} g 1.0".format(np.round(j,3))}
#     evaluator = Evaluator(args,loading=True, path="../data-vans/")
#     energies.append(evaluator.raw_history[len(list(evaluator.raw_history.keys()))-1][-1])
#     np.save("results/xxz/plot/energies14xxz_lowest",energies)

energies = np.load("results/xxz/plot/energies14xxz_lowest.npy")
plt.figure(figsize=(20,20))
ax1 = plt.subplot2grid((2,1),(0,0))
ax2 = plt.subplot2grid((2,1),(1,0))


plt.suptitle("14 qubits\n"+r'$H = g \sum_j \sigma_j^{z} + \sum_j \sigma_{j}^{x} \sigma_{j+1}^{x} +  \sigma_{j}^{y} \sigma_{j+1}^{y} + J  \sigma_{j}^{z} \sigma_{j+1}^{z}$', size=20)
ax1.scatter(js,energies, s=20, color="red", label="lowest energy found (vans)" )
ax1.plot(js, ans, color="black",label="true ground state energy")
ax2.scatter(js,np.abs((energies-ans)/ans), s=30, color="red")
ax2.set_ylabel(r'$\frac{\Delta E}{E_g}$', size=20)
ax2.set_xlabel("J", size=30)
#for a in [ax1, ax2]:#
    #a.set_xticks(np.arange(0,4.6,0.4))
ax1.legend(prop={"size":15})
plt.savefig("results/xxz/plot/xxz14.png")
# # # plt.plot(energies)
# # # plt.show()
