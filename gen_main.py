from vans_gym.solvers import GeneticSolver
import os
import pickle
from tqdm import tqdm
import numpy as np

sols = {}
J=1.21
for dep in [0.1, 0.01]:
    print("dep", dep)
    sol = GeneticSolver(n_qubits= 2, qlr=0.1, qepochs=10, g=1, J=J, noises={"depolarizing":dep}, verbose=1)
    sol.history_circuits=[]
    history_energies=[]
    best_energies_found = []

    gates_index = [sol.number_of_cnots] ## begin with a certain circuit
    gates_index, resolver, energy= sol.run_circuit_from_index(gates_index)
    sol.history_circuits.append(gates_index)
#    sol.current_circuit = gates_index

    for kk in tqdm(range(4)):
        print("\n%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("new iteration: ",kk)
        print("1:" )
        print(sol.give_circuit(gates_index)[0])

        enns = [energy]
        which_block = np.random.choice([0,1], p=[.5,.5])
        if which_block == 0:
            qubit = np.random.choice(sol.n_qubits)
            block_to_insert = sol.resolution_1qubit(qubit)
            insertion_index = np.random.choice(max(1,len(gates_index))) #gives index between \in [0, len(gates_index) )
        else:
            qubits = np.random.choice(sol.n_qubits, 2,replace = False)
            block_to_insert = sol.resolution_2cnots(qubits[0], qubits[1])
            insertion_index = np.random.choice(max(1,len(gates_index))) #gives index between \in [0, len(gates_index) )

        #print(block_to_insert)
        ### optimize the circuit with the block appended. This is tricky since we initialize
        ###  the continuous parameters with the older ones, and the "block ones" close to identity
        circuit, variables = sol.prepare_circuit_insertion(gates_index, block_to_insert, insertion_index) #this either accepts or reject the insertion
        model = sol.initialize_model_insertion(variables) ### initialize the model in the previously optimized parameters & resolution to identity for the block

        gates_index, resolver, energy, accepted = sol.optimize_and_update(gates_index,model, circuit, variables, insertion_index,block_to_insert) #inside, if better circuit is found, saves it.
        del model
        print("2:")
        print(sol.give_circuit(gates_index)[0])
        if accepted:
            print("accepted")
            sol.history_circuits.append(gates_index)
            #### try to kill one qubit unitaries ###
            for k in range(10):
                if len(gates_index)-sol.count_number_cnots(gates_index) > 2:
                    gates_index, resolver, energy, simplified =  sol.kill_one_unitary(gates_index, resolver, energy)
                    print("3: ")
                    print(sol.give_circuit(gates_index)[0])
                    sol.history_circuits.append(gates_index)

            ### simplify the circuit and if the length is changed I run the optimization again
            print("about to simplify")
            simplified_gates_index = sol.simplify_circuit(gates_index)
            if len(simplified_gates_index)<len(gates_index) and len(simplified_gates_index)>0:
                print("actually simplified!: running opt")
                ggates_index, rresolver, eenergy = sol.run_circuit_from_index(simplified_gates_index,hyperparameters=[20,0.01]) #here I don't save the resolver since it's a mess
                print("3:")
                print(sol.give_circuit(ggates_index)[0])

                if energy < sol.lowest_energy_found:
                    sol.lowest_energy_found = energy
                    sol.best_circuit_found = gates_index
                    sol.best_resolver_found = resolver
                    gates_index = ggates_index
                    sol.history_circuits.append(gates_index)
                    resolver = resolver
                    energy = eenergy

        sol.new_resolver = {}
        history_energies.append(sol.lowest_energy_found)
        enns=[]
        #print("energy: ", energy, "... j", j)
    sol.history_energies=history_energies
    sols[str(dep)] = sol
    print(history_energies)
    with open("noise_test/"+str(dep)+'.pickle', 'wb') as handle:
        pickle.dump(v, handle, protocol=pickle.HIGHEST_PROTOCOL)




# for k,v in sols.items():
