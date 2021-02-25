# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# from utilities.circuit_basics import Evaluator
#
#
# # matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
# # matplotlib.rc('font', serif='cm10')
# # matplotlib.rc('text', usetex=True)
# # #
# axinticks=[]
# plt.rcParams.update({'font.size': 45})
#
#

#
# for j in np.linspace(0,10,20):
#
#     evaluator = Evaluator(loading=True, args={"n_qubits":4, "J":j})
#     energies.append(evaluator.raw_history[len(list(evaluator.raw_history.keys()))-1][-1])
#
# plt.figure(figsize=(10,10))
# ax1 = plt.subplot2grid((2,1),(0,0))
# ax2 = plt.subplot2grid((2,1),(1,0))
#
# plt.suptitle("4 qubits\n"+r'$H = -g \sum_j \sigma_j^{z} - J \sum_j \sigma_{j}^{x} \sigma_{j+1}^{x}$', size=20)
# ax1.scatter(js,energies, s=20, color="red", label="lowest energy found (vans)" )
# ax1.plot(js, ans, color="black",label="true ground state energy")
# ax2.scatter(js,np.abs((energies-ans)/ans), s=30, color="red")
# ax2.set_ylabel(r'$\frac{\Delta E}{E_g}$', size=20)
# ax2.set_xlabel("J", size=30)
# ax1.legend(prop={"size":15})
# # plt.show()
# plt.savefig("TFIM/tfim4.png")
# # plt.plot(energies)
# # plt.show()








import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
import sys
from tqdm import tqdm
sys.path[0] = "/home/cooper-cooper/Desktop/vans/"
plt.style.use('results/plots/style.mplstyle')

from utilities.oldev import Evaluator
matplotlib.rc('text', usetex=True)
axinticks=[]

ground = np.genfromtxt('results/TFIM/plot/TFIM4_v2.csv',delimiter=',')
js,ans = ground[:,0], ground[:,1]
energies = []

for j in tqdm(np.linspace(0,10,20)):
    args={"J":j, "n_qubits":4}
    evaluator = Evaluator(loading=True, args=args)
    energies.append(evaluator.raw_history[len(list(evaluator.raw_history.keys()))-1][4])
    np.save("results/TFIM/plot/energiesTFIM4_lowest",energies)

relatives=np.abs((energies-ans)/ans)

#
#energies = []

# ####### TWO AXIS ####
# energies = np.load("results/TFIM/plot/energiesTFIM8.npy")
# plt.figure(figsize=(20,20))
# ax1 = plt.subplot2grid((2,1),(0,0))
# ax2 = plt.subplot2grid((2,1),(1,0))
#
# plt.suptitle("8 qubits\n"+r'$H = -g \sum_j \sigma_j^{z} - J \sum_j \sigma_{j}^{x} \sigma_{j+1}^{x}$')
# ax1.plot(js, ans, color="blue",alpha=0.7,label="Ground state energy")
# ax1.scatter(js,energies, s=300, alpha=1.0, color="red", label="VAns" )
# ax2.scatter(js,np.abs((energies-ans)/ans), s=300, color="red")
# ax2.set_ylabel(r'$\frac{\Delta E}{E_g}$')
# ax2.set_xlabel("J")
#
# ax1.legend(prop={"size":40})
# plt.savefig("results/TFIM/plot/tfim8",format="pdf")



####### SINGLE AXIS ####
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#
# plt.figure(figsize=(20,10))
# ax1 = plt.subplot2grid((1,1),(0,0))
# plt.subplots_adjust(bottom=0.15,left=0.075)
# plt.suptitle(r'$H = -g \sum_j \sigma_j^{z} - J \sum_j \sigma_{j}^{x} \sigma_{j+1}^{x}$',size=60)
# ax1.plot(js, ans, color="blue",alpha=0.7,label=r'$E_{ground}$')
# ax1.scatter(js,energies, s=400, alpha=1.0, color="red", label="VAns" )
# # ax1.scatter(js[0:1],energies[0:1], s=300, color="purple",alpha=1, label=r'$\frac{\Delta E}{E_ground}$')
#
# ax1.set_xlabel("J",size=70)
# ax1.set_yticks([np.round(k,0) for k in np.linspace(np.min(energies), np.max(energies), 4)])
# ax1.tick_params(direction='out', length=6, width=2, colors='black', grid_alpha=0.5,labelsize=40)
#
#
# axins = inset_axes(ax1, width="50%", height="40%", loc=3,borderpad=3)
# axins.scatter(js,relatives, s=300, color="green", alpha=0.7, label=r'$\frac{\Delta E}{E_{ground}}$')
# axins.xaxis.set_visible(False)
#
# axins.set_yticks([np.round(k,2) for k in np.linspace(np.min(relatives), np.max(relatives), 4)])
# axins.yaxis.tick_right()
# axins.tick_params(direction='out', length=3, width=1, colors='r',grid_color='r', grid_alpha=0.5,labelsize=40)
# # axins.set_yticks([], minor=True)
# # labs = [l.get_label() for l in [ax1,axins]]
#
# # ff=p0+p1+p2
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = axins.get_legend_handles_labels()
#
# ax1.legend(lines+lines2, labels+labels2, prop={"size":40})
# plt.savefig("results/TFIM/plot/tfim8",format="pdf")
#
#
#
#
#
#
# #
