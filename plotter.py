import matplotlib.pyplot as plt
import numpy as np
import argparse

#program that plots the results obtained from HPC..

parser = argparse.ArgumentParser()
parser.add_argument("number_run", type=int)
args = parser.parse_args()
number_run = args.number_run


run = "DueDQN/run_"+str(number_run)

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
