import pickle
import numpy as np
import gym
from vans_gym.envs import VansEnvsSeq
from vans_gym.solvers import CirqSolverR, Checker
from tqdm import tqdm

solver = CirqSolverR(n_qubits = 2, observable_name="Ising_",qlr=0.05,qepochs=100)
checker = Checker(solver)

env = VansEnvsSeq(solver,checker=checker, depth_circuit=4)
gates_number = len(solver.alphabet) - solver.n_qubits

energies={}
next_states={}
seqs=[]
for g1 in np.append(np.arange(gates_number), np.array([-1])):
    for g2 in np.append(np.arange(gates_number), np.array([-1])):
        for g3 in np.append(np.arange(gates_number), np.array([-1])):
            for g4 in np.append(np.arange(gates_number), np.array([-1])):
                next_states[str(np.array([g1,g2,g3,g4]))] ={}
for g1 in tqdm(np.arange(gates_number)): #I quit the identity..
    dumm = np.array([-1]*env.depth_circuit)
    s = dumm.copy()
    s[0] = g1
    next_states[str(dumm)][str(int(g1))] = s
    for g2 in np.arange(gates_number):
        ns = env.checker.correct_trajectory([g1,g2])
        for j in np.arange(env.depth_circuit-len(ns)):
            ns = np.append(ns, np.array(-1))
        next_states[str(np.array([g1, -1,-1,-1]))][str(int(g2))] = ns
        for g3 in np.arange(gates_number):
            ns = env.checker.correct_trajectory([g1,g2,g3])
            for j in np.arange(env.depth_circuit-len(ns)):
                ns = np.append(ns, np.array(-1))
            next_states[str(np.array([g1, -1,-1,-1]))][str(int(g3))] = ns

            for g4 in np.arange(gates_number):
                ns = env.checker.correct_trajectory([g1,g2,g3,g4])
                c=0
                for j in np.arange(env.depth_circuit-len(ns)):
                    c+=1
                    ns = np.append(ns, np.array(-1))
                next_states[str(np.array([g1,g2,g3,-1]))][str(int(g4))] = ns
                if c==0:
                    energy = solver.run_circuit(ns, sim_q_state=False)
                    energies[str(np.array(ns))] =energy
