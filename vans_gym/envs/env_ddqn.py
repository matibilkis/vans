import numpy as np
import matplotlib.pyplot as plt
import pickle

import gym
from gym import spaces


class VansEnvsSeq(gym.Env):
    def __init__(self, solver, checker, depth_circuit=8, printing=False):
        """"

        Simple environment that takes the state as sequence.

        depth_circuit: nnumber of gates you put in total (notice when you have more than 2 this checked by parallelization). Maybe leave identities at the end (only one per node).


        The state is considered to be
        [-1 -1 -1 -1 ] at begining.
        [g1 -1 -1 -1]
        [...]
        [g1 g2 ... g_{depth_circuit}] at the end

        checker:: (not included in solver yet). parallelization.

        """
        super(VansEnvsSeq, self).__init__()
        self.solver = solver
        self.checker = checker
        self.n_qubits = solver.n_qubits
        self.depth_circuit = depth_circuit
        self.state_shape = int(self.depth_circuit)


        self.printing = printing
        self.history_final_reward = []
        self.episode = 0

        self.n_actions = len(solver.alphabet)
        self.state = np.ones(self.depth_circuit)*-1.
        self.sequence = []

        self.i_step = 0

    def reset(self):
        self.episode += 1
        self.i_step = 0
        self.state = np.ones(self.depth_circuit)*-1.
        self.sequence = np.array([])
        return self.state.astype(np.float32)

    def check_if_finish(self):
        return len(self.sequence) >= self.depth_circuit or self.i_step>(10*self.depth_circuit)

    def step(self, action, checking=True):
        """the action is an index of the alphabet"""
        self.sequence = np.append(self.sequence,action)
        self.i_step +=1
        if checking:
            try:
                self.sequence = self.checker.correct_trajectory(self.sequence)
            except IndexError:
                pass #this may be due to little number of gates
        self.state[:len(self.sequence)] = self.sequence
        #print(self.state,"ep", self.episode, action, np.random.random())
        ### INSERT CHECKER HERE! ###
        done = self.check_if_finish()
        if done:
            reward = self.solver.run_circuit(self.state, sim_q_state=False)
        else:
            reward = 0.

        if done and self.printing:
            self.history_final_reward = np.append(self.history_final_reward, reward)
            if self.episode % 1 == 0 and not self.in_callback:
                print(f"\n================= Episodes {self.episode} =================\n")
                print("List gates", self.state)
                print("Reward", reward)
                print("Mean reward last 100 episodes", np.mean(self.history_final_reward[-100:]))
        return self.state, reward, done, {}

    def render(self, mode="human"):
        #$fig, ax = plt.subplots(1, 1)
        return #fig
