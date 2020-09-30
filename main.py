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

    info = "\n\n\n\nYou are using GENETIC-VANS: \n"
    info += f"len(n_qubits): {vqe_handler.n_qubits}\n" \
                        f"g: {vqe_handler.g}, \n" \
                        f"J: {vqe_handler.J}\n" \
                        f"qlr: {vqe_handler.lr}\n" \
                        f"qepochs: {vqe_handler.epochs}\n" \
                        f"patience: {vqe_handler.patience}\n" \
                        f"genetic runs: {args.reps}\n"
    print(info)



    indexed_circuit=[vqe_handler.number_of_cnots+k for k in range(vqe_handler.n_qubits,2*vqe_handler.n_qubits)]
    energy, symbols_to_values, training_evolution = vqe_handler.vqe(indexed_circuit) ##very nie 5000, 0.01
    evaluator.add_step(indexed_circuit, symbols_to_values, energy, relevant=True)
    evaluator.lowest_energy = energy

    ### create a mutation M (maybe this word is too fancy)
    M_circuit, M_symbols_to_values, M_idx_to_symbols = iid.randomly_place_almost_identity(indexed_circuit, symbols_to_values)
