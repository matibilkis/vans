import gc
import numpy as np
import sympy
import cirq
import tensorflow_quantum as tfq
from tqdm import tqdm
import tensorflow as tf
from solver import GeneticSolver
import argparse
import os
import pickle

class History:
    def __init__(self,g=None,J=None):
        self.history={}
        self.raw_history = {}
        self.novel_discoveries = {}
        self.hamiltonian_parameters={"g":g,"J":J}
        self.lowest_energy = 0.



    def accept_energy(self, E, kill_one_unitary=False):
        if kill_one_unitary:
            return (E-self.lowest_energy)/np.abs(self.lowest_energy) < 0.01
        else:
            return E < self.lowest_energy



if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--J", type=float, default=1.21)
    parser.add_argument("--n_qubits", type=int, default=3)

    parser.add_argument("--noise", type=bool, default=False)

    args = parser.parse_args()

    sol = GeneticSolver(n_qubits=args.n_qubits, g=1, J=args.J, qlr=.01, qepochs=100, noise=args.noise)
    historial=History(g=sol.g,J=sol.J)

    ### initialize circuit ####
    indices=[sol.number_of_cnots+k for k in range(sol.n_qubits)]
    circuit, symbols, index_to_symbols = sol.give_circuit(indices)

    ##### Compute energy of the first initial proposal ####
    SymbolToValue, energy = sol.compute_energy_first_time(circuit, symbols,[100,0.01])
    historial.history["0"] = [circuit, SymbolToValue, energy]
    historial.lowest_energy = energy
    historial.novel_discoveries[str(len(list(historial.novel_discoveries.keys())))] = [circuit, SymbolToValue, energy]
    historial.raw_history[str(len(list(historial.raw_history.keys())))] = [circuit, SymbolToValue, energy]


    for it in tqdm(range(30)):
        ####### append a block #####
        which_block = np.random.choice([0,1], p=[.5,.5])
        if which_block == 0:
            qubit = np.random.choice(sol.n_qubits)
            block_to_insert = sol.resolution_1qubit(qubit)
            insertion_index = np.random.choice(max(1,len(indices))) #gives index between \in [0, len(gates_index) )
        else:
            qubits = np.random.choice(sol.n_qubits, 2,replace = False)
            block_to_insert = sol.resolution_2cnots(qubits[0], qubits[1])
            insertion_index = np.random.choice(max(1,len(indices))) #gives index between \in [0, len(gates_index) )

        #### check if you can reduce the circuit with this block added:
        NewIndices, New_Idx_To_Symbols, New_SymbolToValue = sol.prepare_circuit_insertion(indices, block_to_insert, insertion_index, SymbolToValue)
        simp_indices, idxToSymbol, SymbolToVal = sol.simplify_circuit(NewIndices, New_Idx_To_Symbols, New_SymbolToValue) #it would be great to have a way to realize that insertion was trivial...RL? :-)
        ### iterate many times circuit simplification
        simp_indices, idxToSymbol, SymbolToVal = sol.reduce_circuit(simp_indices, idxToSymbol, SymbolToVal)
        #iterate between killingone unitary and simplifying the circuit
        simplified, itt = True, 0
        while simplified or itt<10:
            itt+=1
            k_simp_indices, k_symbol_to_val, k_new_energy, k_idx_to_symbols, simplified = sol.kill_one_unitary(indices, SymbolToValue, historial)
            if simplified:
                ### iterate many times circuit simplification
                simp_indices, idxToSymbol, SymbolToVal = sol.reduce_circuit(k_simp_indices, k_idx_to_symbols, k_symbol_to_val)
            else:
                break

        model = sol.initialize_model_insertion(SymbolToVal)
        new_energy = sol.optimize_model_from_indices(simp_indices, model)

        historial.raw_history[str(len(list(historial.raw_history.keys())))] = [sol.give_circuit(simp_indices)[0], SymbolToVal, new_energy]

        if historial.accept_energy(new_energy):
            SymbolToValue = {s:k for s, k in zip(list(SymbolToVal.keys()), model.trainable_variables[0].numpy())}
            historial.lowest_energy = new_energy
            energy = new_energy
            indices=simp_indices

            historial.novel_discoveries[str(len(list(historial.novel_discoveries.keys())))] = [sol.give_circuit(indices)[0], SymbolToValue, energy]

        historial.history[str(it)] = [sol.give_circuit(indices)[0], SymbolToValue,energy]
        del model
        gc.collect()

    if not os.path.exists("results"):
        os.makedirs("results")

    with open("results/noisy"+str(sol.J)+'.pickle', 'wb') as handle:
        pickle.dump(historial, handle, protocol=pickle.HIGHEST_PROTOCOL)
