
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
from tqdm import tqdm
import matplotlib.colors as colors
sys.path[0] = "/home/cooper-cooper/Desktop/vans/"

#
converter = colors.ColorConverter()

plt.style.use("results/plots/style.mplstyle")
from utilities.variational import VQE
from utilities.circuit_basics import *
from utilities.evaluator import Evaluator
matplotlib.rc("text",usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
axinticks=[]

#ground = np.genfromtxt('results/TFIM/TFIM8.csv',delimiter=',')
#energies = np.load("results/TFIM/energiesTFIM8.npy")
#js,ans = ground[:,0], ground[:,1]
#relatives=np.abs((energies-ans)/ans)




marco_hea_2=0.6549782647945612
marco_hea_5=0.42458460441807233

energies = []
J=5.0
#args={"n_qubits":8,"problem_config":{"problem" : "XXZ", "g":1.0, "J": J}, "load_displaying":False, "specific_folder_name":"8Q - J {} g 1.0".format(J)}
#args={"n_qubits":8,"problem_config":{"problem" : "XXZ", "g":1.0, "J": J}, "load_displaying":False, "specific_folder_name":"8Q - J {} g 1.0".format(J)}
bond=1.46
problem_config_load={"problem" : "H2", "geometry": str([('H', (0., 0., 0.)), ('H', (0., 0., bond))]).replace("\'",""), "multiplicity":1, "charge":0, "basis":"sto-3g"}
problem_config={"problem" : "H2", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"}

indi=15
J=np.round(np.loadtxt("results/hea/evolution/TFIM8.csv", delimiter=",")[:,0][indi],3)
ground = np.loadtxt("results/hea/evolution/TFIM8.csv", delimiter=",")[:,1][indi]

problem_config={"problem" : "TFIM", "g":1.0, "J": J}
#args={"n_qubits":4,"problem_config":problem_config_load, "load_displaying":False, "specific_folder_name":"4_bd_0.98"}
args ={"n_qubits":8,"problem_config":problem_config,"specific_folder_name":"8Q - J {} g 1.0".format(J)}
vqe_handler = VQE(n_qubits=args["n_qubits"],problem_config=problem_config)


evaluator = Evaluator(args,loading=True, path="../data-vans/")
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


color1="#D79922"
color2="#4056A1"
color3="#F13C20"
color4="#5D001E"
color5="#8E8D8A"

nparams=[]
cnots=[]
energies=[]
for index in range(len(list(evaluator.evolution.keys()))):
    index_circuit=evaluator.evolution[index][2]
    cnots.append(vqe_handler.count_cnots(index_circuit))
    nparams.append(vqe_handler.count_params(index_circuit))
    energies.append(evaluator.evolution[index][1])

plt.figure(figsize=(20,20))
ax2 = plt.subplot2grid((2,1),(0,0))
ax1 = plt.subplot2grid((2,1),(1,0))
plt.subplots_adjust(bottom=0.15,left=0.18)
# plt.subplots_adjust(bottom=0.15,left=0.075)


CNOTS2 = vqe_handler.count_cnots(vqe_handler.hea_ansatz_indexed_circuit(L=2))
CNOTS5 = vqe_handler.count_cnots(vqe_handler.hea_ansatz_indexed_circuit(L=5))
PARAMS2 = vqe_handler.count_params(vqe_handler.hea_ansatz_indexed_circuit(L=2))
PARAMS5 = vqe_handler.count_params(vqe_handler.hea_ansatz_indexed_circuit(L=5))

ax1.plot(np.ones(len(nparams))*CNOTS2, color="red",linewidth=5, alpha=0.8)
ax1.plot(np.ones(len(nparams))*CNOTS5, color="green",linewidth=5, alpha=0.8)
ax1.plot(np.ones(len(nparams))*PARAMS2, '--',color="red",linewidth=5, alpha=0.8)
ax1.plot(np.ones(len(nparams))*PARAMS5, '--',color="green",linewidth=5, alpha=0.8)


ax1.plot(cnots, label="CNots", color="blue",linewidth=5, alpha=0.5)
ax1.plot(nparams, label="Trainable parameters", color="grey",linewidth=5, alpha=0.8)
ax1.set_xlabel("Accepted modifications",size=70)
ax1.set_ylabel("Circuit \nstructure",size=70)
ax2.plot(energies, alpha=0.8,color=converter.to_rgb(color3), label="VAns")
#ax2.plot(np.ones(len(energies))*compute_ground_energy_1(vqe_handler.observable,vqe_handler.qubits),
#         color="black",alpha=0.8,  linewidth=3, label="Ground state energy")
ax2.plot(np.ones(len(energies))*ground,
         color="black",alpha=0.75, label="Ground state energy")

ax2.plot(np.ones(len(energies))*(ground+marco_hea_2),
         color=converter.to_rgb(color2),alpha=0.75,   label="HEA 2 Layers")

ax2.plot(np.ones(len(energies))*(ground+marco_hea_5),
         color=converter.to_rgb(color1),alpha=0.75,   label="HEA 5 Layers")

ax2.set_yticks(np.round(np.linspace(ground,-8,4),1))
ax1.set_yticks(range(0,81,20))

#ax1.set_xticks(range(0,len(energies)+1,100))

ax2.legend(prop={"size":40},loc=1)
ax1.legend(prop={"size":40},loc=0,borderpad=.4)

ax2.xaxis.set_visible(False)
ax2.set_ylabel("Energy",size=70)
ax1.tick_params(direction='out', length=6, width=2, colors='black', grid_alpha=0.5,labelsize=60)
ax2.tick_params(direction='out', length=6, width=2, colors='black', grid_alpha=0.5,labelsize=60)


plt.savefig("results/hea/evolution/evolution_circuit_tfim{}.pdf".format(J),format="pdf")
