import os

# for k in range(2):
#     os.system("python3 meta_plotter.py --names "+r'"01GreedyPolicy"'+" --run "+str(k))


import matplotlib.pyplot as plt
import numpy as np
import argparse

#program that plots the results obtained from HPC..

for dir in ["RandomPolicy","01GreedyPolicy", "exp-greedy"]:
    for number_run in range(15):
        run = dir+"/run_"+str(number_run)

        try:

            rcum_per_e = np.load(run+"/data_collected/cumulative_reward_per_episode.npy", allow_pickle=True)
            rehist = np.load(run+"/data_collected/reward_history.npy", allow_pickle=True)
            lhist = np.load(run+"/data_collected/loss.npy", allow_pickle=True)
            pt = np.load(run+"/data_collected/pgreedy.npy", allow_pickle=True)


            plt.figure(figsize=(20,20))
            ax1 = plt.subplot2grid((1,2), (0,0))
            ax2 = plt.subplot2grid((1,2), (0,1))
            ax1.plot(pt, alpha=0.6,c="blue", linewidth=1,label="greedy policy")
            ax1.scatter(np.arange(1,len(rehist)+1), rehist, alpha=0.5, s=50, c="black", label="reward")
            ax1.plot(rcum_per_e, alpha=0.6, linewidth=9,c="red",label="cumulative reward")
            ax2.plot(range(len(lhist)), lhist, alpha=0.6, linewidth=1,c="blue",label="critic loss")
            ax1.legend(prop={"size":20})
            ax2.legend(prop={"size":20})
            plt.savefig(run+"/learning_curves.png")
        except Exception:
            print("Error with ",run)
            print("\n")
