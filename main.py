import gc
import numpy as np
import sympy
import cirq
import tensorflow_quantum as tfq
from tqdm import tqdm
import tensorflow as tf

from variational import VQE
from circuit_basics import Evaluator
from idinserter import IdInserter
from simplifier import Simplifier
from unitary_killer import UnitaryMurder


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
    vqe_handler = VQE(n_qubits=args.n_qubits, lr=0.01, epochs=2000, patience=100, random_perturbations=True, verbose=0, g=1, J = args.J)
    evaluator = Evaluator(n_qubits=args.n_qubits)
    iid = IdInserter(n_qubits=args.n_qubits)
    Simp = Simplifier(n_qubits=args.n_qubits)
    killer = UnitaryMurder(n_qubits=args.n_qubits, g=1, J = args.J)

    info = "\n\n\n\nYou are using GENETIC-VANS: \n"
    info += f"len(n_qubits): {vqe_handler.n_qubits}\n" \
                        f"g: {vqe_handler.g}, \n" \
                        f"J: {vqe_handler.J}\n" \
                        f"qlr: {vqe_handler.lr}\n" \
                        f"qepochs: {vqe_handler.epochs}\n" \
                        f"patience: {vqe_handler.patience}\n" \
                        f"genetic runs: {args.reps}\n"
    print(info)

    ### begin with a product ansatz
    indexed_circuit=[vqe_handler.number_of_cnots+k for k in range(vqe_handler.n_qubits,2*vqe_handler.n_qubits)]
    energy, symbol_to_value, training_evolution = vqe_handler.vqe(indexed_circuit) #compute energy

    #add initial info to evaluator
    evaluator.add_step(indexed_circuit, symbol_to_value, energy, relevant=True)
    evaluator.lowest_energy = energy

    ### create a mutation M (maybe this word is too fancy)
    M_indices, M_symbols_to_values, M_idx_to_symbols = iid.randomly_place_almost_identity(indexed_circuit, symbol_to_value)

    ### simplify the circuit as much as possible
    Sindices, Ssymbols_to_values, Sindex_to_symbols = Simp.reduce_circuit(M_indices, M_symbols_to_values, M_idx_to_symbols)

    ## compute the energy of the mutated-simplified circuit [Note 1]
    MSenergy, MSsymbols_to_values, _ = vqe_handler.vqe(Sindices)

    if evaluator.accept_energy(MSenergy):
        indexed_circuit, symbol_to_value, index_to_symbols = Sindices, MSsymbols_to_values, Sindex_to_symbols
        # unitary slaughter: delete as many 1-qubit gates as possible, as long as the energy doesn't go up (we allow %1 increments per iteration)
        cnt=0
        reduced=True
        lmax=len(indexed_circuit)
        while reduced and cnt < lmax:
            indexed_circuit, symbol_to_value, index_to_symbols, energy, reduced = killer.unitary_slaughter(indexed_circuit, symbol_to_value, index_to_symbols)
            indexed_circuit, symbol_to_value, index_to_symbols = Simp.reduce_circuit(indexed_circuit, symbol_to_value, index_to_symbols)
            cnt+=1

        evaluator.add_step(indexed_circuit, symbol_to_value, energy)

    print("current energy: ", energy)
    print(vqe_handler.give_unitary(indexed_circuit,symbol_to_value))
    print("\n")
### [Note 1]: Even if the circuit gets simplified to the original one, it's harmless to compute the energy again since i) you give another try to the optimization, ii) we have the EarlyStopping and despite of the added noise, it's supossed the seeds are close to optima.
