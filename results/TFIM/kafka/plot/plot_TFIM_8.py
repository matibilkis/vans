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

ground = np.genfromtxt('results/TFIM/TFIM8.csv',delimiter=',')
energies = np.load("results/TFIM/energiesTFIM8.npy")
js,ans = ground[:,0], ground[:,1]
relatives=np.abs((energies-ans)/ans)

#
#energies = []
# for j in tqdm(np.arange(0,4.7,0.1)):
#
#     evaluator = Evaluator(loading=True, args={"n_qubits":8, "J":np.round(j,2)})
#     energies.append(evaluator.raw_history[len(list(evaluator.raw_history.keys()))-1][-1])
#     np.save("results/TFIM/plot/energiesTFIM8_lowest",energies)
    #
    # args={"n_qubits":14,"problem_config":{"problem" : "XXZ", "g":1.0, "J": j}, "load_displaying":False,"specific_folder_name":"14Q - J {} g 1.0".format(np.round(j,3))}
    # evaluator = Evaluator(args,loading=True, path="../data-vans/")
    # vansenergies.append(evaluator.evolution[evaluator.get_best_iteration()][4])


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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


converter = colors.ColorConverter()


color1="#D79922"
color2="#4056A1"
color3="#F13C20"
color4="#5D001E"
color5="#8E8D8A"


#
# color1="#D79922"
# color2="#4056A1"
# color3="#F13C20"
# color4="#5D001E"
# color5="#8E8D8A"
#
# plt.figure(figsize=(30,20))
# ax1 = plt.subplot2grid((2,1),(1,0))
# ax2 = plt.subplot2grid((2,1),(0,0))
#
# plt.subplots_adjust(bottom=0.15,left=0.1)
# plt.suptitle(r'$H = \sum_i \;\sigma_i^x \sigma^x_{i+1} + \sigma^{y}_i \sigma^y_{i+1} + \Delta \sigma^z_i \sigma^z_{i+1} + g \;\sum_i \sigma_i^{z}$',size=55)
# ax1.scatter(np.arange(-3,5.1,0.1),energies, marker="h",s=250, alpha=1, color="black", label="VAns")
# ax1.plot(js,np.array(ge), color=converter.to_rgb(color3),alpha=1, label="ground energy")
# ax1.plot(js,np.array(ge1),color=converter.to_rgb(color2), alpha=1, label="first excited")
# ax1.plot(js,np.array(ge2), '--',color=converter.to_rgb(color1),alpha=1, label="second excited")
#
# ax1.set_xlabel(r'$\Delta$',size=70)
# ax1.set_yticks([np.round(k,0) for k in np.linspace(np.min(energies), np.max(energies), 4)])
# ax1.tick_params(direction='out', length=6, width=2, colors='black', grid_alpha=0.5,labelsize=40)
#
# energies=np.array(energies)
# ax2.plot(js,np.abs((energies-np.array(ge))/ge), color=converter.to_rgb(color3),alpha=1, label="ground energy")
# # ax2.plot(js,np.abs((energies-np.array(ge1))/ge1),color=converter.to_rgb(color2), alpha=1, label="first excited")
# # ax2.plot(js,np.abs((energies-np.array(ge2))/ge2), '--',color=converter.to_rgb(color1),alpha=1, label="second excited")
# #
# # #
# ind1=int(0.3*len(energies))
# ind2=ind1+20
# axins = inset_axes(ax1, width="70%", height="70%", borderpad=3, loc=3, bbox_to_anchor=(.1, .05, .8, .45), bbox_transform=ax1.transAxes)
# axins.scatter(np.arange(-3,5.1,0.1)[ind1:ind2],energies[ind1:ind2], marker="h",s=250, alpha=1, color="black", label="VAns")
# axins.plot(js[ind1:ind2],np.array(ge)[ind1:ind2], color=converter.to_rgb(color3),alpha=1, label="ground energy")
# axins.plot(js[ind1:ind2],np.array(ge1)[ind1:ind2],color=converter.to_rgb(color2), alpha=1, label="first excited")
# axins.plot(js[ind1:ind2],np.array(ge2)[ind1:ind2], '--',color=converter.to_rgb(color1),alpha=1, label="second excited")
#











plt.figure(figsize=(20,10))
ax1 = plt.subplot2grid((1,1),(0,0))
plt.subplots_adjust(bottom=0.15,left=0.075)
plt.suptitle(r'$H = -g \sum_j \sigma_j^{z} - J \sum_j \sigma_{j}^{x} \sigma_{j+1}^{x}$',size=60)
ax1.plot(js, ans,color=converter.to_rgb(color3),alpha=0.7,label=r'$E_{ground}$')
ax1.scatter(js,energies, s=400, alpha=1.0, color="black", label="VAns" )
# ax1.scatter(js[0:1],energies[0:1], s=300, color="purple",alpha=1, label=r'$\frac{\Delta E}{E_ground}$')

ax1.set_xlabel("J",size=70)
ax1.set_yticks([np.round(k,0) for k in np.linspace(np.min(energies), np.max(energies), 4)])
ax1.tick_params(direction='out', length=6, width=2, colors='black', grid_alpha=0.5,labelsize=40)


axins = inset_axes(ax1, width="50%", height="40%", loc=3,borderpad=3)
axins.plot(js,relatives,  color=converter.to_rgb(color5), alpha=0.7, label=r'$\frac{\Delta E}{E_{ground}}$')
axins.xaxis.set_visible(False)

axins.set_yticks([np.round(k,2) for k in np.linspace(np.min(relatives), np.max(relatives), 4)])
axins.yaxis.tick_right()
axins.tick_params(direction='out', length=3, width=1, colors='black',grid_color='r', grid_alpha=0.5,labelsize=40)
# axins.set_yticks([], minor=True)
# labs = [l.get_label() for l in [ax1,axins]]

# ff=p0+p1+p2
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = axins.get_legend_handles_labels()

ax1.legend(lines+lines2, labels+labels2, prop={"size":40})
plt.savefig("results/TFIM/tfim8qbits.pdf",format="pdf")






#