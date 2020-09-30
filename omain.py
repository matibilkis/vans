import gc
import numpy as np
import sympy
import cirq
import tensorflow_quantum as tfq
from tqdm import tqdm
import tensorflow as tf

from variational import VQE
from circuit_basics import Evaluator
import argparse
import os
import pickle
from datetime import datetime


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--J", type=float, default=0.0)
    parser.add_argument("--n_qubits", type=int, default=3)

    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--names", type=str, default="obj")
    parser.add_argument("--folder_result", type=str, default="results")


    args = parser.parse_args()


    begin = datetime.now()
    vqe_handler = VQE(n_qubits=args.n_qubits, lr=0.01, epochs=2000, patience=100, random_perturbations=True, verbose=0, g=1, J = args.J) ##very nie 5000, 0.01
    evaluator = Evaluator(n_qubits=args.n_qubits)


    info = "\n\n\n\nYou are using GENETIC-VANS: \n"
    info += f"len(n_qubits): {vqe_handler.n_qubits}\n" \
                        f"g: {vqe_handler.g}, \n" \
                        f"J: {vqe_handler.J}\n" \
                        f"qlr: {vqe_handler.qlr}\n" \
                        f"qepochs: {vqe_handler.qepochs}\n" \
                        f"patience: {vqe_handler.patience}\n" \
                        f"genetic runs: {args.reps}\n"
    print(info)



    indexed_circuit=[sol.number_of_cnots+k for k in range(sol.n_qubits,2*sol.n_qubits)]
    energy, symbols_to_values, training_evolution = continuous_optimization_handler.vqe(indexed_circuit)
    evaluator.add_step(indexed_circuit, symbols_to_values, energy, relevant=True)









###############################
###############################
    for iteration in tqdm(range(args.reps)):

        print("Iteration "+str(iteration-1)+" has finished after: ", str(datetime.now()-begin) )
        begin = datetime.now()

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

            indexed_circuit, index_to_symbols, symbol_to_value = sol.simplify_kill_simplify(indexed_circuit_proposal, index_to_symbols_proposal, trained_symols_to_val, historial.lowest_energy)

            historial.lowest_energy = new_energy
            historial.novel_discoveries[str(len(list(historial.novel_discoveries.keys())))] = [sol.give_circuit(indexed_circuit)[0], index_to_symbols, new_energy]

            energy = new_energy #to save in the history..
        del model
        gc.collect()
        historial.history[str(iteration)] = [sol.give_circuit(indexed_circuit)[0], index_to_symbols, energy]

    print("iteration: ", iteration)
    print("energy of current circuit: ",energy)
    print("\n")
    print(cirq.resolve_parameters(historial.history[str(iteration)][0],historial.history[str(iteration)][1]) )
    print(historial.history[str(iteration)][-1])

    print("\n")
    print("\n")
    print("RAW")
    print(cirq.resolve_parameters(historial.raw_history[str(iteration)][0],historial.raw_history[str(iteration)][1]) )
    print(historial.raw_history[str(iteration)][-1])


    if not os.path.exists(args.folder_result):
        os.makedirs(args.folder_result)

    with open(args.folder_result+"/"+args.names+'.pickle', 'wb') as handle:
        pickle.dump(historial, handle, protocol=pickle.HIGHEST_PROTOCOL)
