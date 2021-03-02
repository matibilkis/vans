import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
sys.path[0] = "../../"

from tqdm import tqdm
# from utilities.evaluator import Evaluator
# from utilities.variational import VQE
# from utilities.misc import compute_ground_energy_1
import os

plt.style.use('results/plots/style.mplstyle')

matplotlib.rc("text",usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

js = np.arange(0,2.2,.2)
energies=[]
path="../data-vans/"
longens=[]
longranejs=np.linspace(0,2.2,100)





jass=[]
energies=[]
ans=[]
# for j in tqdm(js):
#     try:
#         args={"n_qubits":4,"problem_config":{"problem" : "TFIM", "g":1.0, "J": np.round(j,2)}, "load_displaying":False}
#         evaluator = Evaluator(args,loading=True, path=path)
#         energies.append(evaluator.raw_history[len(list(evaluator.raw_history.keys()))-1][4])
#         jass.append(j)
#
#         vqe_handler = VQE(n_qubits=args["n_qubits"],problem_config=args["problem_config"])
#         obs = vqe_handler.observable
#         eigs = compute_ground_energy_1(obs, vqe_handler.qubits)
#         ans.append(eigs[0])
#
#         np.save("results/TFIM/energies4TFIM_lowest",energies)
#         np.save("results/TFIM/jsenergies4TFIM_lowest",jass)
#         np.save("results/TFIM/ans4TFIM",ans)
#
#     except Exception as e:
#         print(e)
#         pass
#
# for jj in longranejs:
#     args={"n_qubits":4,"problem_config":{"problem" : "TFIM", "g":1.0, "J": jj}, "load_displaying":False}
#     vqe_handler = VQE(n_qubits=args["n_qubits"],problem_config=args["problem_config"])
#     obs = vqe_handler.observable
#     eigs = compute_ground_energy_1(obs, vqe_handler.qubits)
#     longens.append(eigs[0])
# np.save("results/TFIM/bruteforce_TFIM4",longens)


# from utilities.misc import compute_ground_energy_1
# from utilities.variational import VQE

import matplotlib.colors as colors
converter = colors.ColorConverter()
#


longens=np.load("results/TFIM/datos/4qubits/bruteforce_TFIM4.npy")
energies=np.load("results/TFIM/datos/4qubits/energies4TFIM_lowest.npy")
jss=np.load("results/TFIM/datos/4qubits/jsenergies4TFIM_lowest.npy")
ans= np.load("results/TFIM/datos/4qubits/ans4TFIM.npy")



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

plt.subplots_adjust(bottom=0.15,left=0.175)
plt.suptitle(r'$H = -g \sum_j \sigma_j^{z} - J \sum_j \sigma_{j}^{x} \sigma_{j+1}^{x}$',size=60)
ax1.scatter(jss,energies,s=400, alpha=1, color=converter.to_rgb(color2), label="VAns")
ax1.plot(longranejs,np.array(longens), color=converter.to_rgb(color1),alpha=.7, label="ground energy")

ax2.set_xlabel(r'$J$',size=70)
ax1.set_yticks([np.round(k,0) for k in np.linspace(np.min(energies), np.max(energies), 4)])
ax1.tick_params(direction='out', length=12, width=4, colors='black', grid_alpha=0.5)
energies=np.array(energies)
ax2.plot(jss,np.abs((energies-np.array(ans))/ans), color=converter.to_rgb(color3),alpha=0.84, label=r'$\frac{\Delta E}{E_{ground}}$')


ax1.xaxis.set_visible(False)
ax2.set_yticks([np.round(k,9) for k in np.linspace(0., np.max(np.abs((energies-np.array(ans))/ans)), 4)])
ax2.tick_params(direction='out', length=12, width=4, colors='black', grid_alpha=0.5)
ax2.set_ylabel("Relative error",size=70)
ax1.set_ylabel("Energy",size=70)
#

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
#
###incremento x ---> se va a la izquierda. Incremento y ---> se va para arriba
ax1.legend(lines2+lines, labels2+labels, prop={"size":45}, loc=1, borderaxespad=.2)
plt.savefig("results/TFIM/TFIM4qbits.pdf",format="pdf")
