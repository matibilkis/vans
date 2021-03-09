import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
from tqdm import tqdm
import matplotlib.colors as colors
#
converter = colors.ColorConverter()
sys.path[0] = "/home/cooper-cooper/Desktop/vans/"
plt.style.use('results/plots/style.mplstyle')

# from utilities.evaluator import Evaluator
matplotlib.rc("text",usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
axinticks=[]
from utilities.evaluator import Evaluator
from utilities.variational import VQE


#
energies = []
ans=[]
js=list(np.arange(0,2,0.1))
jsp = list(np.arange(2,2.8,.1))
js = js + jsp
js = np.array(js)

# for j in tqdm(js):
#
#     args={"n_qubits":8,"problem_config":{"problem" : "TFIM", "g":1.0, "J": j}, "load_displaying":False}
#     evaluator = Evaluator(args,loading=True, path="../data-vans/")
#     energies.append(evaluator.evolution[evaluator.get_best_iteration()][1])
#
#     vqe_handler = VQE(n_qubits=args["n_qubits"],problem_config=args["problem_config"])
#     ans.append(vqe_handler.lower_bound_energy)
#
#     np.save("results/TFIM/datos/ans8",ans)
#     np.save("results/TFIM/datos/energies8",energies)

ans=np.load("results/TFIM/datos/ans8.npy")
energies=np.load("results/TFIM/datos/energies8.npy")

# ####### TWO AXIS ####
# energies = np.load("results/TFIM/energiesTFIM8.npy")
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
# plt.savefig("results/TFIM/tfim8",format="pdf")



####### SINGLE AXIS ####
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


converter = colors.ColorConverter()


color1="#D79922"
color2="#4056A1"
color3="#F13C20"
color4="#5D001E"
color5="#8E8D8A"


# jss=js
# plt.figure(figsize=(30,20))
# ax2 = plt.subplot2grid((2,1),(1,0))
# ax1 = plt.subplot2grid((2,1),(0,0))
#
# plt.subplots_adjust(bottom=0.15,left=0.1)
# plt.suptitle(r'$H = -g \sum_j \sigma_j^{z} - J \sum_j \sigma_{j}^{x} \sigma_{j+1}^{x}$',size=60)
# ax1.scatter(jss,energies, marker="h",s=250, alpha=1, color="black", label="VAns")
# ax1.plot(longranejs,np.array(longens), color=converter.to_rgb(color3),alpha=1, label="ground energy")
#
#
# ax1.set_yticks([np.round(k,0) for k in np.linspace(np.min(energies), np.max(energies), 4)])
# ax1.tick_params(direction='out', length=6, width=2, colors='black', grid_alpha=0.5,labelsize=40)
# energies=np.array(energies)
# ax2.plot(jss,np.abs((energies-np.array(ans))/ans), color=converter.to_rgb(color5),alpha=0.84, label=r'$\frac{\Delta E}{E_{ground}}$')

# print(np.max(np.abs((energies-np.array(ans))/ans)))
# ax2.set_yticks([np.round(k,9) for k in np.linspace(0., np.max(np.abs((energies-np.array(ans))/ans)), 4)])
plt.figure(figsize=(20,30))
ax2 = plt.subplot2grid((2,1),(1,0))
ax1 = plt.subplot2grid((2,1),(0,0))

relatives = np.abs((np.array(energies)-np.array(ans))/np.array(ans))

plt.subplots_adjust(bottom=0.15,left=0.18)

plt.suptitle(r'$H = -g \sum_j \sigma_j^{z} - J \sum_j \sigma_{j}^{x} \sigma_{j+1}^{x}$',size=60)


ax1.plot(js, ans,color=converter.to_rgb(color1),alpha=0.7,label=r'$E_{ground}$')
ax1.scatter(js,energies, s=400, alpha=1.0, color=converter.to_rgb(color2), label="VAns" )
# ax1.scatter(js[0:1],energies[0:1], s=300, color="purple",alpha=1, label=r'$\frac{\Delta E}{E_ground}$')


ax1.tick_params(direction='out', length=6, width=2, colors='black', grid_alpha=0.5)#),labelsize=40)
ax2.plot(js,np.log10(relatives),  color=converter.to_rgb(color3), alpha=0.7, label=r'$\frac{\Delta E}{E_{ground}}$')
ax2.set_yticks(np.round(np.log10(relatives)[0::2]),2)
ax2.set_xticks(np.round(np.array(list(js[0::4]) + [2.7]),2))
labs=[]
labs1=[]
for k in np.log10(relatives)[0::2]:
    labs.append("{}".format(np.round(10**k,4)))
    labs1.append("{}".format(10**k))

print(labs)
print(labs1)
print(np.log10(relatives)[0::2])
ax2.set_yticklabels(labs)

ax2.set_xlabel(r'$J$',size=70)
ax1.tick_params(direction='out', length=6, width=2, colors='black', grid_alpha=0.5)#,labelsize=40)
ax1.xaxis.set_visible(False)

ax2.tick_params(direction='out', length=6, width=2, colors='black', grid_alpha=0.5)#,labelsize=40)
ax2.set_ylabel("Relative error",size=70)
ax1.set_ylabel("Energy",size=70)
#

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
#
###incremento x ---> se va a la izquierda. Incremento y ---> se va para arriba
ax1.legend(lines2+lines, labels2+labels, prop={"size":35}, loc=0, borderaxespad=.2)



plt.savefig("results/TFIM/tfim8qbits_.pdf",format="pdf")






#
