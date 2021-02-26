import sys
sys.path[0] = "/home/cooper-cooper/Desktop/vans/"

from utilities.evaluator import Evaluator
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
from utilities.variational import VQE
import matplotlib
import matplotlib.colors as colors
converter = colors.ColorConverter()


plt.style.use('results/plots/style.mplstyle')
matplotlib.rc("text",usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

energies=[]
iterations=[]
fcis=[]
js=np.linspace(0.5,2.3,16)

color1="#D79922"
color2="#4056A1"
color3="#F13C20"
color4="#5D001E"
color5="#8E8D8A"

# jsn=np.linspace(np.min(js),np.max(js),100)
# fcisn=[]
# for bond in tqdm(jsn):
#
#     problem_config ={"problem" : "H2", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"}
#     args={"n_qubits":8,"problem_config":problem_config, "load_displaying":False,"specific_folder_name":"4_bd_{}".format(bond)}
#     vqe_handler = VQE(n_qubits=4,noise_config={}, problem_config=problem_config,
#                         return_lower_bound=True)
#
#     fcisn.append(vqe_handler.lower_bound_energy)
#


# for bond in tqdm(js):
#     problem_config ={"problem" : "H2", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"}
#     args={"n_qubits":8,"problem_config":problem_config, "load_displaying":False,"specific_folder_name":"4_bd_{}".format(bond)}
#     evaluator = Evaluator(args,loading=True, path="../data-vans/")
#     energies.append(evaluator.raw_history[len(list(evaluator.raw_history.keys()))-1][-1])
#     iterations.append(len(list(evaluator.raw_history.keys())))
#         #VQE module, in charge of continuous optimization
#     vqe_handler = VQE(n_qubits=4,noise_config={}, problem_config=problem_config,
#                         return_lower_bound=True)
#     fcis.append(vqe_handler.lower_bound_energy)
#
#
# os.makedirs("results/molecular/h2/data_plot",exist_ok=True)
# np.save("results/molecular/h2/data_plot/vansenergies",energies)
# np.save("results/molecular/h2/data_plot/fcis",fcis)
# np.save("results/molecular/h2/data_plot/iterations",iterations)
# np.save("results/molecular/h2/data_plot/fcisn",fcisn)
# np.save("results/molecular/h2/data_plot/jsn",jsn)
# #

energies=np.load("results/molecular/h2/data_plot/vansenergies.npy")
fcis=np.load("results/molecular/h2/data_plot/fcis.npy")
iterations=np.load("results/molecular/h2/data_plot/iterations.npy")
fcisn=np.load("results/molecular/h2/data_plot/fcisn.npy")
jsn=np.load("results/molecular/h2/data_plot/jsn.npy")

plt.figure(figsize=(30,30))
# plt.subplots_adjust(bottom=0.15,left=0.075, wspace=0.1)
s=23
for k in range(3):

    ax = plt.subplot2grid((3,1),(k,0))
    if k==0:
        ax.plot(jsn,fcisn,color=converter.to_rgb(color2),alpha=0.7,label="E(FCI)")
        ax.scatter(js,energies,marker='o',s=400,color=converter.to_rgb(color1),alpha=0.7,label="E(VAns)")
        # ax.set_ylabel("Energy",size=s)
        ax.set_xticks([])


        ax.set_yticks([np.round(k,2) for k in np.linspace(np.min(energies), np.max(energies), 4)])
        # ax.tick_params(direction='out', length=6, width=2, colors='black', grid_alpha=0.5,labelsize=40)
        ax.legend(prop={"size":30})
    elif k==1:
        ax.plot(js,energies-fcis)
        ax.scatter(js, energies-fcis,marker='o',color=converter.to_rgb(color1),s=400,alpha=0.7,label="E(Vans)-E(FCI)")
        # ax.set_ylabel("",size=s)
        ax.yaxis.set_label_position("left")
        ax.plot(js,np.ones(len(js))*0.0016,'--',color="black",alpha=0.75,label="Chemical accuracy")
        ax.set_xticks([])
        ax.set_yticks([np.round(k,4) for k in np.linspace(0, 0.002, 6)])
        ax.tick_params(direction='out', length=6, width=2, colors='black', grid_alpha=0.5)
        # ax.yaxis.tick_right()
        ax.legend(prop={"size":30})

    else:
        for j,b in zip(js,iterations):
            ax.bar(j,b,alpha=0.85,align="center",width=0.05)#np.std(js[0:1]))
        ax.set_ylabel("Iterations to converge",size=2*s)
        ax.yaxis.set_label_position("left")
        # wh=[]
        # jj=[np.round(k,2) for k in np.linspace(np.min(js), np.max(js), 10)]
        jj=np.round(js,2)
        #for k in jj:
    #        wh.append(np.squeeze(np.where(js==k)))
        ax.set_xticks(jj)
        ax.set_xticklabels(jj)
        ax.set_xlabel("Bond Lenght [Å]")

        # ax.set_yticks([np.arange(1,101,25)])
        # ax.yaxis.tick_right()


    # ax.legend()
plt.savefig("results/molecular/h2/H2.pdf",format="pdf")