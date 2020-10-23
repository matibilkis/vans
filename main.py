import os
import gc
import numpy as np
import sympy
import cirq
import tensorflow_quantum as tfq
from tqdm import tqdm
import tensorflow as tf
import json
import argparse
import pickle
from datetime import datetime


from utilities.variational import VQE
from utilities.circuit_basics import Evaluator
from utilities.idinserter import IdInserter
from utilities.simplifier import Simplifier
from utilities.unitary_killer import UnitaryMurder



if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--J", type=float, default=0.0)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--n_qubits", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--names", type=str, default="obj")
    parser.add_argument("--folder_result", type=str, default="results")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--qepochs", type=int, default=10**4)
    parser.add_argument("--qlr", type=float, default=0.01)
    parser.add_argument("--problem", type=str, default="TFIM")
    parser.add_argument('--noise_model', type=json.loads, default='--{}')

    args = parser.parse_args()

    args = parser.parse_args()

    begin = datetime.now()
    #VQE in charge of continuous optimization

    vqe_handler = VQE(n_qubits=args.n_qubits, lr=args.qlr, epochs=args.qepochs, patience=100, random_perturbations=True, verbose=args.verbose, g=args.g, J = args.J, noise_model=args.noise_model, problem=args.problem)

    start = datetime.now()
    info = f"len(n_qubits): {vqe_handler.n_qubits}\n" \
                        f"g: {vqe_handler.g}, \n" \
                        f"noise: {args.noise_model}\n"\
                        f"J: {vqe_handler.J}\n" \
                        f"qlr: {vqe_handler.lr}\n" \
                        f"qepochs: {vqe_handler.epochs}\n" \
                        f"patience: {vqe_handler.patience}\n" \
                        f"genetic runs: {args.reps}\n"
    print("\n"*3+info)

    #Evaluator keeps a record of the circuit and accepts or not certain configuration
    evaluator = Evaluator(args, info=info)
    evaluator.displaying +=info
    #IdInserter appends to a given circuit an identity resolution
    iid = IdInserter(n_qubits=args.n_qubits)

    #Simplifier reduces gates number as much as possible while keeping same expected value of target hamiltonian
    Simp = Simplifier(n_qubits=args.n_qubits)

    #UnitaryMuerder is in charge of evaluating changes on the energy while setting apart one (or more) parametrized gates. If
    killer = UnitaryMurder(vqe_handler)


    ### begin with a product ansatz
    indexed_circuit=[vqe_handler.number_of_cnots+k for k in range(vqe_handler.n_qubits,2*vqe_handler.n_qubits)]
    energy, symbol_to_value, training_evolution = vqe_handler.vqe(indexed_circuit) #compute energy

    #add initial info to evaluator
    evaluator.add_step(indexed_circuit, symbol_to_value, energy, relevant=True)
    evaluator.lowest_energy = energy

    for iteration in range(args.reps):
        relevant=False
        ### create a mutation M (maybe this word is too fancy)
        M_indices, M_symbols_to_values, M_idx_to_symbols = iid.place_almost_identity(indexed_circuit, symbol_to_value)

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
            relevant=True
        evaluator.add_step(indexed_circuit, symbol_to_value, energy, relevant=relevant)

        to_print="\nIteration #{}\nTime since beggining:{}\n ".format(iteration, datetime.now()-start)+str("**"*20)+"\n"+str(str("*"*20))+"\ncurrent energy: {}\n\n{}\n\n".format(energy,vqe_handler.give_unitary(indexed_circuit,symbol_to_value))
        to_print+="\n\n"
        print(to_print)
        evaluator.displaying +=to_print

        ## save results of iteration.
        evaluator.save_dicts_and_displaying()
### [Note 1]: Even if the circuit gets simplified to the original one, it's harmless to compute the energy again since i) you give another try to the optimization, ii) we have the EarlyStopping and despite of the added noise, it's supossed the seeds are close to optima.
