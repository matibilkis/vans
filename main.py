import os
import gc
import numpy as np
import sympy
import cirq

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_quantum as tfq
from tqdm import tqdm
import tensorflow as tf
import json
import argparse
import pickle
from datetime import datetime

from utilities.variational import VQE
from utilities.evaluator import Evaluator
from utilities.idinserter import IdInserter
from utilities.simplifier import Simplifier
from utilities.unitary_killer import UnitaryMurder
from utilities.misc import scheduler_selector_temperature, scheduler_parameter_perturbation_wall #this outputs always 10 for now.


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n_qubits", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--path_results", type=str, default="../data-vans/")
    parser.add_argument("--specific_name", type=str, default="")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--qepochs", type=int, default=10**4)
    parser.add_argument("--qlr", type=float, default=0.01)
    parser.add_argument("--training_patience", type=int, default=200)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--problem_config", type=json.loads, default='{}')
    parser.add_argument("--noise_config", type=json.loads, default='{}')
    parser.add_argument("--acceptance_percentage", type=float, default=0.01)
    parser.add_argument("--reduce_acceptance_percentage",type=float,default=0.0)
    parser.add_argument("--return_lower_bound", type=int, default=0)
    parser.add_argument("--show_tensorboarddata",type=int, default=0)
    parser.add_argument("--rate_iids_per_step",type=float,default=1)
    parser.add_argument("--selector_temperature",type=float,default=10.0)
    parser.add_argument("--initialization",type=str,default="hea")
    parser.add_argument("--wait_to_get_back",type=int,default=25) #notice there's some correspondence with the annealing in the perturbations_wall

    args = parser.parse_args()
    reduce_acceptance_percentage=[False,True][int(args.reduce_acceptance_percentage)]


    begin = datetime.now()

    #VQE module, in charge of continuous optimization
    vqe_handler = VQE(n_qubits=args.n_qubits, lr=args.qlr, epochs=args.qepochs, verbose=args.verbose,
                        noise_config=args.noise_config, problem_config=args.problem_config,
                        patience=args.training_patience, return_lower_bound=[True, False][args.return_lower_bound], optimizer=args.optimizer)

    start = datetime.now()

    info = f"len(n_qubits): {vqe_handler.n_qubits}\n" \
                        f"noise: {args.noise_config}\n"\
                        f"qlr: {vqe_handler.lr}\n" \
                        f"qepochs: {vqe_handler.epochs}\n" \
                        f"patience: {vqe_handler.patience}\n" \
                        f"genetic runs: {args.reps}\n" \
                        f"optimizer: {args.optimizer}\n" \
                        f"acceptance_percentage runs: {args.acceptance_percentage}\n" \
                        f"reduce_acceptance_percentage: {reduce_acceptance_percentage}\n" \
                        f"temperature_iid_resolution_selector: {args.selector_temperature}\n" \
                        f"rate_iids_per_step: {args.rate_iids_per_step}\n" \
                        f"initialization: {args.initialization}\n" \
                        f"Wait to get back to favorite: {args.wait_to_get_back} \n"\
                        f"problem_info: {args.problem_config}\n"

    if vqe_handler.problem_nature == "chemical":
        accuracy_to_end = vqe_handler.lower_bound_energy + 0.0016 #chemical accuracy
    else:
        accuracy_to_end = vqe_handler.lower_bound_energy

    #Evaluator keeps a record of the circuit and accepts or not certain configuration
    evaluator = Evaluator(vars(args), info=info, path=args.path_results, acceptance_percentage=args.acceptance_percentage, accuracy_to_end=accuracy_to_end, reduce_acceptance_percentage=reduce_acceptance_percentage)

    evaluator.displaying["information"]+=info

    if args.show_tensorboarddata == 1:
        vqe_handler.tensorboarddata = evaluator.directory

    #IdInserter appends to a given circuit an identity resolution
    iid = IdInserter(n_qubits=len(vqe_handler.qubits))
    iid.selector_temperature = args.selector_temperature

    #Simplifier reduces gates number as much as possible while keeping same expected value of target hamiltonian
    Simp = Simplifier(n_qubits=len(vqe_handler.qubits))

    #UnitaryMuerder is in charge of evaluating changes on the energy while setting apart one (or more) parametrized gates. If
    killer = UnitaryMurder(vqe_handler, noise_config=args.noise_config)

    if args.initialization == "hea":
        indexed_circuit = vqe_handler.hea_ansatz_indexed_circuit(L=8)
    elif args.initialization == "separable":
        indexed_circuit=[vqe_handler.number_of_cnots+k for k in range(vqe_handler.n_qubits,2*vqe_handler.n_qubits)]
    elif args.initialization == "xz":
        indexed_circuit=[]
        for i in range(len(vqe_handler.qubits)):
            indexed_circuit.append(vqe_handler.number_of_cnots+ vqe_handler.n_qubits +i)
            indexed_circuit.append(vqe_handler.number_of_cnots+i)
    else:
        raise NameError("Please choose your initial ansatz!")

    print("beggining to train!")
    energy, symbol_to_value, training_evolution = vqe_handler.vqe(indexed_circuit) #compute energy
    #add initial info to evaluator

    to_print="\nIteration #{}\nTime since beggining:{}\n best energy: {}\ncurrent_energy {}\n lower_bound: {}".format(0, datetime.now()-start, energy, energy, evaluator.accuracy_to_end)
    print(to_print)
    evaluator.displaying["information"]+=to_print

    evaluator.add_step(indexed_circuit, symbol_to_value, energy, relevant=True)
    evaluator.lowest_energy = energy

    for iteration in range(1,args.reps+1):
        relevant=False

        ### create a mutation M (maybe this word is too fancy); we add (probably more than one) identity resolution
        iid.selector_temperature=scheduler_selector_temperature(energy, evaluator.lowest_energy, when_on=args.selector_temperature)
        M_indices, M_symbols_to_values, M_idx_to_symbols = iid.place_identities(indexed_circuit, symbol_to_value, rate_iids_per_step= args.rate_iids_per_step)

        ### simplify the circuit as much as possible
        Sindices, Ssymbols_to_values, Sindex_to_symbols = Simp.reduce_circuit(M_indices, M_symbols_to_values, M_idx_to_symbols)

        ## compute the energy of the mutated-simplified circuit [Note 1]
        MSenergy, MSsymbols_to_values, _ = vqe_handler.vqe(Sindices, symbols_to_values=Ssymbols_to_values, parameter_perturbation_wall=scheduler_parameter_perturbation_wall(its_without_improvig=evaluator.its_without_improvig))

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

        to_print="\nIteration #{}\nTime since beggining:{}\n best energy: {}\ncurrent energy: {}\n lower_bound: {}".format(iteration, datetime.now()-start, evaluator.lowest_energy,energy, evaluator.accuracy_to_end)
        print(to_print)
        evaluator.displaying["information"]+=to_print

        evaluator.save_dicts_and_displaying()

        if evaluator.if_finish_ok is True:
            print("HOMEWORK DONE! \nBeers on me ;)")
            break

        if evaluator.its_without_improvig == args.wait_to_get_back:
            print("Getting back to favorite, it's been already {} iterations".format(args.wait_to_get_back))
            _, energy, indices, resolver, _, _ =  evaluator.evolution[evaluator.get_best_iteration()]
            evaluator.its_without_improvig = 0



### [Note 1]: Even if the circuit gets simplified to the original one, it's harmless to compute the energy again since i) you give another try to the optimization, ii) we have the EarlyStopping and despite of the added noise, it's supossed the seeds are close to optima.
