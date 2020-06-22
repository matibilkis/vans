import numpy as np
import matplotlib.pyplot as plt
import pickle

import gym
from gym import spaces
import pennylane as qml


def projector(ket):
    n = len(ket)
    proj = np.zeros((n,n))
    for ind1, i in enumerate(ket):
        for ind2, j in enumerate(ket):
            proj[ind1, ind2] = i*np.conjugate(j)
    return proj


def append_gate(alphabet, index):
    # let's say the parametrized gates are only rotations of 1 free param (how to )
    if "params" in list(alphabet[str(index)].keys()):
        if alphabet[str(index)]["gate"] == qml.Rot:
            params = alphabet[str(index)]["params"]
            return alphabet[str(index)]["gate"](params[0], params[1],params[2], wires=alphabet[str(index)]["wires"])
        else:
            return alphabet[str(index)]["gate"](alphabet[str(index)]["params"][0], wires=alphabet[str(index)]["wires"])
    else:
        return alphabet[str(index)]["gate"](wires=alphabet[str(index)]["wires"])


class VansEnv(gym.Env):
    def __init__(self, n_qubits, maximum_number_of_gates=15):
        super(VansEnv, self).__init__()
        self.n_qubits = n_qubits
        self.maximum_number_of_gates = maximum_number_of_gates

        self.state_indexed = np.array([])

        with open('alphabet_w.pickle', 'rb') as alphabet:
            self.alphabet = pickle.load(alphabet)

        self.n_actions = len(self.alphabet)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(np.array([-1] * 2 * 2**n_qubits),
                                            np.array([1] * 2 * 2**n_qubits),
                                            dtype=np.float32)

        sq = 1 / np.sqrt(3)
        w_state = np.array([0, sq, sq, 0, sq, 0, 0, 0])
        self.W = projector(w_state)
        self.reward_history = np.array([])

    def projector(self, ket):
        n = len(ket)
        proj = np.zeros((n, n))
        for ind1, i in enumerate(ket):
            for ind2, j in enumerate(ket):
                proj[ind1, ind2] = i * np.conjugate(j)
        return proj

    def reset(self):
        """
        the observation must be a numpy array (??)

        !!!** Not necessarily, notice that Luckasz selects at random, so in principle
        it should be enough to give the sequence of gates done.
        """
        self.state_indexed = np.array([])
        self.reward_history = np.array([])
        return np.array([self.state_indexed]).astype(np.float32)

    def check_if_finish(self):
        # np.count_nonzero(self.state_indexed,self.alphabet.CNOTS_indexes)
        if len(self.state_indexed) > self.maximum_number_of_gates:
            return True
        else:
            return False

    def step(self, action):
        """importantly, action is an integer between 0 and len(self.alphaber)-1 """
        self.state_indexed = np.append(self.state_indexed, action)
        done = self.check_if_finish()
        reward = self.reward()
        self.reward_history = np.append(self.reward_history, reward)
        info = {}
        return self.state_indexed.astype(np.float32), reward, done, info  # maybe instaed of

    def reward(self):
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev)
        def circuit(state_indexed):
            for ind in state_indexed:
                append_gate(self.alphabet, int(ind.val))
            return qml.expval(qml.Hermitian(self.W, wires=[0, 1, 2]))

        return circuit(self.state_indexed)

    def render(self, mode="human"):
        fig, ax = plt.subplots(1, 1)

        return fig
