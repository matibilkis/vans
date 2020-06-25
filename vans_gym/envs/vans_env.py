import numpy as np
import matplotlib.pyplot as plt
import pickle

import gym
from gym import spaces


class VansEnv(gym.Env):
    def __init__(self, solver, maximum_number_of_gates=9, bandit=False):
        super(VansEnv, self).__init__()
        self.solver = solver
        self.n_qubits = solver.n_qubits
        self.maximum_number_of_gates = maximum_number_of_gates
        self.bandit=bandit

        if self.bandit:
            self.state_indexed =  np.array( [0,1,2,3,4,5,4,6])
        else:
            self.state_indexed=np.array([])
        self.n_actions = len(solver.alphabet)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(np.array([0] * 2**self.n_qubits),
                                            np.array([1] * 2**self.n_qubits),
                                            dtype=np.float32) #notice M. changed -1 to 0, given that we are using the probs.

        self.reward_history = np.array([])
        self.quantum_state = np.array([0. for _ in range(2**self.n_qubits)])
        self.fidelity = np.inf

        self.episode = -1

    def reset(self):
        self.episode += 1

        if self.bandit:
            self.state_indexed = np.array( [0,1,2,3,4,5,4,6]) # np.array( [0,1,2,3,4,5,4,6,7]) is the optimal
        else:
            self.state_indexed=np.array([])
        self.reward_history = np.array([])
        self.quantum_state = np.array([0. for _ in range(2 ** self.n_qubits)])

        print("==================== Episode {} ====================".format(self.episode))

        return self.quantum_state

    def check_if_finish(self, reward):
        # np.count_nonzero(self.state_indexed,self.alphabet.CNOTS_indexes)

        return len(self.state_indexed) > self.maximum_number_of_gates or reward == 1

    def step(self, action):
        self.state_indexed = np.append(self.state_indexed, action)

        self.fidelity, self.quantum_state = self.solver.run_circuit(self.state_indexed)
        reward = self.reward()

        self.reward_history = np.append(self.reward_history, reward)
        info = {}

        done = self.check_if_finish(reward)

        if done:
            print("List gates", self.state_indexed)
            print("Reward", reward)

        return self.quantum_state, reward, done, info

    def reward(self):
        return self.fidelity

    def render(self, mode="human"):
        fig, ax = plt.subplots(1, 1)

        return fig
