import numpy as np
import matplotlib.pyplot as plt
import pickle

import gym
from gym import spaces


class VansEnv(gym.Env):
    def __init__(self, solver, depth_circuit=9, training_env=True, state_as_sequence=True, printing=True):
        """If state_as_sequence False, state is taken as the quantum state."""

        super(VansEnv, self).__init__()
        self.solver = solver
        self.n_qubits = solver.n_qubits
        self.depth_circuit = depth_circuit

        self.training_env = training_env
        self.state_as_sequence = state_as_sequence
        self.printing = printing

        self.state_indexed = np.array([])

        self.n_actions = len(solver.alphabet)
        self.action_space = spaces.Discrete(self.n_actions)
        self.state_sequence = np.ones(self.depth_circuit)*-1.

        if self.state_as_sequence is True:
            self.observation_space = spaces.Box(np.array([-self.n_actions] * self.depth_circuit),
                                                np.array([self.n_actions] * self.depth_circuit),
                                                dtype=np.float32)
            self.state_shape = int(self.depth_circuit)
        else:
            self.observation_space = spaces.Box(np.array([0] * 2**self.n_qubits),
                                                np.array([1] * 2**self.n_qubits),
                                                dtype=np.float32)
            self.state_shape = int(2**self.n_qubits)

        # For callbacks and printing
        self.reward_history = np.array([])
        self.history_final_reward = np.array([])

        self.quantum_state = np.array([0. for _ in range(2**self.n_qubits)])
        self.quantum_state[0] = 1.

        # self.target_function = np.inf

        self.episode = -1
        self.i_step = 0
        self.max_reward_so_far = 0
        self.in_callback = False

    def reset(self):
        self.episode += 1

        self.state_indexed=np.array([])
        self.i_step = 0

        self.reward_history = np.array([])
        self.quantum_state = np.array([0. for _ in range(2 ** self.n_qubits)])
        self.quantum_state[0] = 1.
        self.state_sequence = np.ones(self.depth_circuit)*-1.

        if self.state_as_sequence:
            self.state = self.state_sequence
            self.state = self.state.astype(np.float32)
        else:
            self.state = self.quantum_state
        return self.state

    def check_if_finish(self):
        if self.state_as_sequence:
            return len(self.state_indexed) >= self.depth_circuit
        else:
            return (self.max_reward_so_far < self.current_reward) or (len(self.state_indexed) >= self.depth_circuit)

        # np.count_nonzero(self.state_indexed,self.alphabet.CNOTS_indexes)
        #     self.max_reward_so_far = reward
        #     return True and self.training_env

    def step(self, action):
        self.state_indexed = np.append(self.state_indexed, action)

        if self.state_as_sequence:
            self.state_sequence[:(len(self.state_indexed))] = self.state_indexed  # fill the -1s with actions...
            state = self.state_sequence.astype(np.float32)
            done = self.check_if_finish()
            if done:
                reward, quantum_state = self.solver.run_circuit(self.state_indexed)
            else:
                reward = 0.
        else:  # state as quantum state.
            self.current_reward, state = self.solver.run_circuit(self.state_indexed)
            done = self.check_if_finish()
            if done:
                self.max_reward_so_far = max(self.max_reward_so_far, self.current_reward)
                reward = self.current_reward
            else:
                reward = 0.
        # if not self.in_callback:
        #     self.i_step += 1
        #     self.reward_history = np.append(self.reward_history, reward)

        if done and self.printing:
            self.history_final_reward = np.append(self.history_final_reward, reward)
            if self.episode % 1 == 0 and not self.in_callback:
                print(f"\n================= Episodes {self.episode} =================\n")
                print("List gates", self.state_indexed)
                if self.state_as_sequence is False:
                    print("state: ", self.state)
                print("Final quantum state", self.quantum_state)
                print("Reward", reward)
                print("Mean reward last 100 episodes", np.mean(self.history_final_reward[-100:]))

        return state, reward, done, {}

    # def reward(self):
    #     r = self.target_function
    #     if len(self.state_indexed) >= 2 and self.state_indexed[-1] == self.state_indexed[-2]:
    #         r = -1
    #     return r

    def render(self, mode="human"):
        fig, ax = plt.subplots(1, 1)

        return fig
