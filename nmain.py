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

def diff(u_1, u_2, cnots_simplified = False, numpy_type=True):
    ui = cirq.unitary(u_1)
    uf = cirq.unitary(u_2)
    if cnots_simplified:
        return np.sum(np.abs((ui - uf)[:,0]))
    else:
        return np.sum(np.abs((ui - uf)))




if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--J", type=float, default=0.0)
    parser.add_argument("--n_qubits", type=int, default=3)

    parser.add_argument("--noise", type=bool, default=False)
    parser.add_argument("--reps", type=int, default=15)

    args = parser.parse_args()


    sol = GeneticSolver(n_qubits=args.n_qubits, g=1, J=args.J, qlr=.01, qepochs=5000, noise=args.noise)
    historial=History(g=sol.g,J=sol.J)


    ### initialize circuit ####
    indices=[sol.number_of_cnots+k for k in range(sol.n_qubits)]
    circuit, symbols, index_to_symbols = sol.give_circuit(indices)

    ##### Compute energy of the first initial proposal ####
    SymbolToValue, energy,h = sol.compute_energy_first_time(circuit, symbols,[5000,0.01])

    historial.history["0"] = [circuit, SymbolToValue, energy]
    historial.lowest_energy = energy
    historial.novel_discoveries[str(len(list(historial.novel_discoveries.keys())))] = [circuit, SymbolToValue, energy]
    historial.raw_history[str(len(list(historial.raw_history.keys())))] = [circuit, SymbolToValue, energy]

    for iteration in range(args.reps):
        print(iteration, energy, "\n")
        print(sol.give_unitary(indices, SymbolToValue))
        print("\n")
        ##notice that even appending a dummy gate will be helpful,
        #since it can be interpreted as "another chance" for this configuration in the continuous optimization.

        which_block = np.random.choice([0,1], p=[.5,.5])

        if which_block == 0:
            qubit = np.random.choice(sol.n_qubits)
            block_to_insert = sol.resolution_1qubit(qubit)
            insertion_index = np.random.choice(max(1,len(indices))) #gives index between \in [0, len(gates_index) )
        else:
            qubits = np.random.choice(sol.n_qubits, 2,replace = False)
            block_to_insert = sol.resolution_2cnots(qubits[0], qubits[1])
            insertion_index = np.random.choice(max(1,len(indices))) #gives index between \in [0, len(gates_index) )

        #### append the block at the corresponding, keeping the values of other parameters ###
        NewIndices, New_Idx_To_Symbols, New_SymbolToValue = sol.prepare_circuit_insertion(indices, block_to_insert, insertion_index, SymbolToValue,init_params="ori", ep=0.1)
        #### simplify the new circuit as much as possible (without killing any gate) ###
        simp_indices, simp_idx_toSymb, simp_smb_to_val = sol.reduce_circuit(NewIndices, New_Idx_To_Symbols, New_SymbolToValue)

        #### optimize the reduced circuit
        model = sol.initialize_model_insertion(simp_smb_to_val)
        new_energy, h = sol.optimize_model_from_indices(simp_indices, model)
        historial.raw_history[str(len(list(historial.raw_history.keys())))] = [sol.give_circuit(simp_indices)[0],simp_smb_to_val, new_energy]


        ### if the energy is lowered, accept this circuit
        if historial.accept_energy(new_energy):
            trained_symols_to_val = {s:k for s, k in zip(list(simp_smb_to_val.keys()), model.trainable_variables[0].numpy())}
            #but first try to simplify again: for this we iterate between killing one qubit unitaries that don't decrease the energy too much and reduce the circuit
            #notice reduce_circuit iterates many times the function simplify_circuit
            simp_indices, simp_idx_toSymb, simp_smb_to_val = sol.simplify_kill_simplify(simp_indices, simp_idx_toSymb, trained_symols_to_val, historial)
            SymbolToValue = {s:k for s, k in zip(list(simp_smb_to_val.keys()), model.trainable_variables[0].numpy())}

            historial.lowest_energy = new_energy
            energy = new_energy
            indices=simp_indices
            historial.novel_discoveries[str(len(list(historial.novel_discoveries.keys())))] = [sol.give_circuit(indices)[0], SymbolToValue, energy]
        del model
        gc.collect()

    if not os.path.exists("results"):
        os.makedirs("results")

    with open("results/noisy"+str(sol.J)+'.pickle', 'wb') as handle:
        pickle.dump(historial, handle, protocol=pickle.HIGHEST_PROTOCOL)
