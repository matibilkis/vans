import gc
import numpy as np
import sympy
import cirq
import tensorflow_quantum as tfq
from tqdm import tqdm
import tensorflow as tf
from solver import Solver, History
import argparse
import os
import pickle


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


    sol = Solver(n_qubits=args.n_qubits, g=1, J=args.J, qlr=.01, qepochs=5000, noise=args.noise, patience=100)
    historial=History(g=sol.g,J=sol.J)

    indexed_circuit=[sol.number_of_cnots+k for k in range(sol.n_qubits,2*sol.n_qubits)]
    ### maybe add some Rz here ? ###

    ### entangling gates ###
    for q in range(sol.n_qubits-1):
        indexed_circuit.append(sol.cnots_index[str([q,q+1])])


    circuit, symbols, index_to_symbols = sol.give_circuit(indexed_circuit)
    symbol_to_value, energy, h = sol.compute_energy_first_time(circuit, symbols,[10000,0.001]) ##very nie 5000, 0.01

    historial.history["0"] = [circuit, symbol_to_value, energy]
    historial.lowest_energy = energy
    historial.novel_discoveries[str(len(list(historial.novel_discoveries.keys())))] = [circuit, symbol_to_value, energy]
    historial.raw_history[str(len(list(historial.raw_history.keys())))] = [circuit, symbol_to_value, energy]

    for iteration in tqdm(range(args.reps)):
        print(iteration, energy, "\n")

        which_block = np.random.choice([0,1], p=[.5,.5])
        insertion_index = np.random.choice(max(1,len(indexed_circuit)))

        if which_block == 0:
            qubit = np.random.choice(sol.n_qubits)
            block_to_insert = sol.resolution_1qubit(qubit)
        else:
            qubits = np.random.choice(sol.n_qubits, 2,replace = False)
            block_to_insert = sol.resolution_2cnots(qubits[0], qubits[1])


        indexed_circuit_proposal, index_to_symbols_proposal, symbol_to_value_proposal = sol.prepare_circuit_insertion(indexed_circuit, block_to_insert, insertion_index, symbol_to_value,init_params="ori", ep=0.1)
        #### simplify the new circuit as much as possible (without killing any gate) ###
        indexed_circuit_proposal, index_to_symbols_proposal, symbol_to_value_proposal = sol.reduce_circuit(indexed_circuit_proposal, index_to_symbols_proposal, symbol_to_value_proposal)

        #### optimize the reduced circuit
        model = sol.initialize_model_insertion(symbol_to_value_proposal)
        new_energy, h = sol.optimize_model_from_indices(indexed_circuit_proposal, model)

        historial.raw_history[str(len(list(historial.raw_history.keys())))] = [sol.give_circuit(indexed_circuit_proposal)[0], symbol_to_value_proposal , new_energy]

        ### if the energy is lowered, accept this circuit! but first try to kill unitaries and reduce it as much as possible.
        #Notice we do so comparing with the lowest energy found
        if historial.accept_energy(new_energy, noise=sol.noise):
            trained_symols_to_val = {s:k for s, k in zip(list(symbol_to_value_proposal.keys()), model.trainable_variables[0].numpy())}

            indexed_circuit, index_to_symbols, symbol_to_value = sol.simplify_kill_simplify(indexed_circuit_proposal, index_to_symbols_proposal, symbol_to_value_proposal, historial.lowest_energy)

            historial.lowest_energy = new_energy
            historial.novel_discoveries[str(len(list(historial.novel_discoveries.keys())))] = [sol.give_circuit(indexed_circuit)[0], index_to_symbols, new_energy]

            energy = new_energy #to save in the history..
        del model
        gc.collect()
        historial.history[str(iteration)] = [sol.give_circuit(indexed_circuit)[0], index_to_symbols, energy]



    if not os.path.exists("results"):
        os.makedirs("results")

    with open("results/"+sol.name_obj+'.pickle', 'wb') as handle:
        pickle.dump(historial, handle, protocol=pickle.HIGHEST_PROTOCOL)
