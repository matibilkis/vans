import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
sys.path[0] = "/home/cooper-cooper/Desktop/vans/"

from tqdm import tqdm
# from utilities.evaluator import Evaluator
# from utilities.variational import VQE
# from utilities.misc import compute_ground_energy_1
import os

plt.style.use('results/plots/style.mplstyle')
matplotlib.rc("text",usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

js = np.arange(0,2.25,.25)
jss=[]
energies=[]
# # path ="/data/uab-giq/scratch/matias/data-vans/"
path="../data-vans/"
# longens=[]
# for jj in np.linspace(0,2.25,100):
#     args={"n_qubits":4,"problem_config":{"problem" : "XXZ", "g":1.0, "J": jj}, "load_displaying":False}
#     vqe_handler = VQE(n_qubits=args["n_qubits"],problem_config=args["problem_config"])
#     obs = vqe_handler.observable
#     eigs = compute_ground_energy_1(obs, vqe_handler.qubits)
#     longens.append(eigs[0])
# np.save("results/xxz/bruteforce_xxz4",longens)
#
# for j in tqdm(js):
#     try:
#
#         args={"n_qubits":4,"problem_config":{"problem" : "XXZ", "g":1.0, "J": j}, "load_displaying":False}
#         evaluator = Evaluator(args,loading=True, path=path)
#         energies.append(evaluator.raw_history[len(list(evaluator.raw_history.keys()))-1][4])
#         np.save("results/xxz/energies4xxz_lowest",energies)
#         np.save("results/xxz/jsenergies4xxz_lowest",jss)
#         jss.append(j)
#     except Exception as e:
#         print(e)
#         pass

#
# from utilities.misc import compute_ground_energy_1
# from utilities.variational import VQE

# ans=[]
# for j in tqdm(js):
#     args={"n_qubits":4,"problem_config":{"problem" : "XXZ", "g":1.0, "J": j}, "load_displaying":False}
#     vqe_handler=VQE(n_qubits=4,problem_config=args["problem_config"], noise_config={})
#     ans.append(compute_ground_energy_1(vqe_handler.observable, vqe_handler.qubits))
# ans = np.array(ans)[:,0]
# np.save("results/xxz/ans4",ans)



import matplotlib.colors as colors
converter = colors.ColorConverter()
#


longens=np.load("results/xxz/data/4qubits/bruteforce_xxz4.npy")
ans = np.load("results/xxz/data/4qubits/ans4.npy")
energies=np.load("results/xxz/data/4qubits/energies4xxz_lowest.npy")
jss=np.load("results/xxz/data/4qubits/jsenergies4xxz_lowest.npy")



axinticks=[]
plt.rcParams.update({'font.size': 45})

converter = colors.ColorConverter()

####### SINGLE AXIS ####
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


color1="#D79922"
color2="#4056A1"
color3="#F13C20"
color4="#5D001E"
color5="#8E8D8A"

plt.figure(figsize=(20,30))
ax2 = plt.subplot2grid((2,1),(1,0))
ax1 = plt.subplot2grid((2,1),(0,0))

plt.subplots_adjust(bottom=0.15,left=0.15)

plt.suptitle(r'$H = \sum_i \;\sigma_i^x \sigma^x_{i+1} + \sigma^{y}_i \sigma^y_{i+1} + \Delta \sigma^z_i \sigma^z_{i+1} + g \;\sum_i \sigma_i^{z}$',size=55)
ax1.scatter(js,energies, marker="h",s=250, alpha=1, color="black", label="VAns")
ax1.plot(np.linspace(0,2.25,100),np.array(longens), color=converter.to_rgb(color3),alpha=1, label="ground energy")

ax2.set_xlabel(r'$\Delta$',size=70)
ax1.set_yticks([np.round(k,0) for k in np.linspace(np.min(energies), np.max(energies), 4)])
ax1.tick_params(direction='out', length=6, width=2, colors='black', grid_alpha=0.5)
energies=np.array(energies)
ax2.plot(js,np.abs((energies-np.array(ans))/ans), color=converter.to_rgb(color5),alpha=0.84, label=r'$\frac{\Delta E}{E_{ground}}$')


ax1.xaxis.set_visible(False)
ax2.set_yticks([np.round(k,9) for k in np.linspace(0., np.max(np.abs((energies-np.array(ans))/ans)), 4)])
ax2.tick_params(direction='out', length=6, width=2, colors='black', grid_alpha=0.5)
ax2.set_ylabel("Relative error",size=70)
ax1.set_ylabel("Energy",size=70)
#

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
#
###incremento x ---> se va a la izquierda. Incremento y ---> se va para arriba
ax1.legend(lines2+lines, labels2+labels, prop={"size":45}, loc=1, borderaxespad=0.1)
plt.savefig("results/xxz/xxz4qbits.pdf",format="pdf")
