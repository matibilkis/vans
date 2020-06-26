import numpy as np
import matplotlib.pyplot as plt
import pickle

import gym
from gym import spaces


class VansEnv(gym.Env):
    def __init__(self, solver, maximum_number_of_gates=15):
        super(VansEnv, self).__init__()
        self.solver = solver
        self.n_qubits = solver.n_qubits
        self.maximum_number_of_gates = maximum_number_of_gates

        self.state_indexed = np.array([])

        self.n_actions = len(solver.alphabet)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(np.array([-1] * 2**self.n_qubits),
                                            np.array([1] * 2**self.n_qubits),
                                            dtype=np.float32)


        self.reward_history = np.array([])
        self.history_final_reward = np.array([])
        self.quantum_state = np.array([0. for _ in range(2**self.n_qubits)])
        self.fidelity = np.inf

        self.episode = -1
        self.i_step = 0

    def reset(self):
        """
        the observation must be a numpy array (??)

        !!!** Not necessarily, notice that Luckasz selects at random, so in principle
        it should be enough to give the sequence of gates done.
        """
        self.episode += 1
        self.i_step = 0
        self.state_indexed = np.array([])
        self.reward_history = np.array([])
        self.quantum_state = np.array([0. for _ in range(2 ** self.n_qubits)])

        return self.quantum_state

    def check_if_finish(self, reward):
        # np.count_nonzero(self.state_indexed,self.alphabet.CNOTS_indexes)
        return len(self.state_indexed) > self.maximum_number_of_gates \
               or reward == 1 \
               or (len(self.state_indexed) > 1 and self.state_indexed[-2] == self.state_indexed[-1])

    def step(self, action):
        self.i_step += 1
        self.state_indexed = np.append(self.state_indexed, action)

        self.fidelity, self.quantum_state = self.solver.run_circuit(self.state_indexed)
        reward = self.reward()

        self.reward_history = np.append(self.reward_history, reward)
        info = {}

        done = self.check_if_finish(reward)

        if not done:
            reward = 0

        if done:
            self.history_final_reward = np.append(self.history_final_reward, reward)
            if self.episode % 50 == 0:
                print("\n==================== Episode {} ====================\n".format(self.episode))
                print("List gates", self.state_indexed)
                print("Reward", reward)
                print("Mean reward 100 episodes", np.mean(self.history_final_reward[-100:]))

        return self.quantum_state, reward, done, info

    def reward(self):
        r = self.fidelity
        if len(self.state_indexed) >= 2 and self.state_indexed[-1] == self.state_indexed[-2]:
            r = -1
        return r

    def render(self, mode="human"):
        fig, ax = plt.subplots(1, 1)

        return fig
