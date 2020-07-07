import numpy as np
import matplotlib.pyplot as plt
import pickle

import gym
from gym import spaces


class VansEnv(gym.Env):
    def __init__(self, solver, maximum_number_of_gates=9, training_env=True, state_as_sequence=False ,mdp_length=None, printing=True, penalty_repeated=False):
        """mdp_length: how many steps you want the episode to last. If set to 1, you have a bandit problem."""

        super(VansEnv, self).__init__()
        self.solver = solver
        self.n_qubits = solver.n_qubits
        self.maximum_number_of_gates = maximum_number_of_gates
        self.mdp_length = mdp_length
        self.training_env = training_env
        self.state_as_sequence = state_as_sequence
        self.printing = printing
        self.penalty_repeated = penalty_repeated #notice this is strongly dependent on how we choose the alphabet..

        if self.mdp_length != None:
            if self.solver.name == "PennylaneSolver":
                self.optimal_sequence =  np.array( [0,1,2,3,4,5,4,6,7]) #np.array( [0,1,2,3,4,5,4,6,7])
                self.state_indexed = self.optimal_sequence[:-self.mdp_length]

                self.maximum_number_of_gates = 9
            elif self.solver.name == "CirqSolver":
                self.optimal_sequence =  np.array([0,1,2,3,4,5,4,6,5,6,7]) #np.array([0,1,2,3,4,5,4,6,5,6,7])
                self.state_indexed = self.optimal_sequence[:-self.mdp_length]
                self.maximum_number_of_gates = 10
        else:
            self.state_indexed=np.array([])
        self.n_actions = len(solver.alphabet)
        self.action_space = spaces.Discrete(self.n_actions)
        self.state_sequence = np.ones(self.maximum_number_of_gates)*-1.

        if self.state_as_sequence is True:
            self.observation_space = spaces.Box(np.array([-self.n_actions] * self.maximum_number_of_gates),
                                                np.array([self.n_actions] *self.maximum_number_of_gates ),
                                                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(np.array([0] * 2**self.n_qubits),
                                                np.array([1] * 2**self.n_qubits),
                                                dtype=np.float32)

        self.reward_history = np.array([])
        self.history_final_reward = np.array([])
        self.quantum_state = np.array([0. for _ in range(2**self.n_qubits)])
        self.quantum_state[0] = 1.

        self.target_function = np.inf

        self.episode = -1
        self.i_step = 0
        self.max_reward_so_far = 0
        self.in_callback = False

    def reset(self):
        self.episode += 1
        if self.mdp_length != None:
            if self.solver.name == "PennylaneSolver":
                self.state_indexed = self.optimal_sequence[:-self.mdp_length]
            elif self.solver.name == "CirqSolver":
                self.state_indexed = self.optimal_sequence[:-self.mdp_length]
        else:
            self.state_indexed=np.array([])
        self.i_step = 0

        self.reward_history = np.array([])
        self.quantum_state = np.array([0. for _ in range(2 ** self.n_qubits)])
        self.quantum_state[0] = 1.
        self.state_sequence = np.ones(self.maximum_number_of_gates)*-1.

        if self.state_as_sequence:
            self.state = self.state_sequence/self.n_actions
            self.state = self.state.astype(np.float32)
        else:
            self.state = self.quantum_state
        return self.state

    def check_if_finish(self, reward):
        # np.count_nonzero(self.state_indexed,self.alphabet.CNOTS_indexes)
        # if self.max_reward_so_far < reward:
        #     self.max_reward_so_far = reward
        #     return True and self.training_env
        return len(self.state_indexed) >= self.maximum_number_of_gates

    def step(self, action):
        self.state_indexed = np.append(self.state_indexed, action)
        self.target_function, self.quantum_state = self.solver.run_circuit(self.state_indexed)
        reward = self.reward()

        if not self.in_callback:
            self.i_step += 1
            self.reward_history = np.append(self.reward_history, reward)

        self.state_sequence[:(len(self.state_indexed))] = self.state_indexed
        done = self.check_if_finish(reward)

        if self.state_as_sequence:
            self.state = self.state_sequence/self.n_actions
            self.state = self.state.astype(np.float32)
        else:
            self.state = self.quantum_state
        if not done:
            reward = 0.#

        if done and self.printing:
            self.history_final_reward = np.append(self.history_final_reward, reward)
            if self.episode % 1 == 0 and not self.in_callback:
                print("\n============ Episode {} ============\n".format(self.episode))
                print("List gates", self.state_indexed)
                if self.state_as_sequence is False:
                    print("state: ", self.state)
                print("Final quantum state", self.quantum_state)
                print("Reward", reward)
                print("Mean reward last 100 episodes", np.mean(self.history_final_reward[-100:]))

        return self.state, reward, done, {}

    def reward(self):
        r = self.target_function
        if len(self.state_indexed) >= 2 and self.state_indexed[-1] == self.state_indexed[-2] and self.penalty_repeated: #we should figure out a way to include CNOTS here (minimum 3..)
            r = -1
        return r

    def render(self, mode="human"):
        fig, ax = plt.subplots(1, 1)

        return fig
