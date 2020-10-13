import cirq
import numpy as np
import sympy
import pickle
import os
from glob import glob

class Basic:
    def __init__(self, n_qubits=3, testing=False):
        """
        n_qubits: number of qubits on your ansatz
        testing: this is inherited by other classes to ease the debugging.
        """
        self.n_qubits = n_qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.lower_bound_Eg = -20*self.n_qubits

        #### keep a register on which integers corresponds to which CNOTS, target or control.
        self.indexed_cnots = {}
        self.cnots_index = {}
        count = 0
        for control in range(self.n_qubits):
            for target in range(self.n_qubits):
                if control != target:
                    self.indexed_cnots[str(count)] = [control, target]
                    self.cnots_index[str([control,target])] = count
                    count += 1
        self.number_of_cnots = len(self.indexed_cnots)

        self.testing=testing


    def append_to_circuit(self, ind, circuit, params, index_to_symbols):
        """
        ind: integer describing the gate to append to circuit
        circuit: cirq object that describes the quantum circuit
        params: a list containing the symbols appearing in the circuit so far
        index_to_sybols: tells which symbol corresponds to i^{th} item in the circuit (useful for simplifier)
        """

        #### add CNOT
        if ind < self.number_of_cnots:
            control, target = self.indexed_cnots[str(ind)]
            circuit.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
            if isinstance(index_to_symbols,dict):
                index_to_symbols[len(list(index_to_symbols.keys()))] = []
                return circuit, params, index_to_symbols
            else:
                return circuit, params

        #### add rz #####
        elif 0 <= ind - self.number_of_cnots  < self.n_qubits:
            qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
            for par, gate in zip(range(1),[cirq.rz]):
                new_param = "th_"+str(len(params))
                params.append(new_param)
                circuit.append(gate(sympy.Symbol(new_param)).on(qubit))
                index_to_symbols[len(list(index_to_symbols.keys()))] = new_param
                return circuit, params, index_to_symbols

        #### add rx #####
        elif self.n_qubits <= ind - self.number_of_cnots  < 2*self.n_qubits:
            qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
            for par, gate in zip(range(1),[cirq.rx]):
                new_param = "th_"+str(len(params))
                params.append(new_param)
                circuit.append(gate(sympy.Symbol(new_param)).on(qubit))
                index_to_symbols[len(list(index_to_symbols.keys()))] = new_param
            return circuit, params, index_to_symbols


    def give_circuit(self, lista):
        """
        retrieves circuit (cirq object), with a list of each continuous parameter (symbols) and dictionary index_to_symbols giving the symbols for each position (useful for simplifier)

        lista: list of integers in [0, 2*n + n(n-1)), with n = self.number_qubits. Each integer describes a possible unitary (CNOT, rx, rz)
        """
        circuit, symbols, index_to_symbols = [], [], {}
        for k in lista:
            circuit, symbols, index_to_symbols = self.append_to_circuit(k,circuit,symbols, index_to_symbols)
        circuit = cirq.Circuit(circuit)
        return circuit, symbols, index_to_symbols

    def give_unitary(self,idx, res):
        """
        a shortcut to resolve parameters.

        idx: sequence of integers encoding the gates
        res: parameters dictionary
        """
        return cirq.resolve_parameters(self.give_circuit(idx)[0], res)


class Evaluator(Basic):
    def __init__(self, args, info=None, loading=False):
        super(Evaluator, self).__init__(n_qubits=args.n_qubits)
        self.raw_history = {}
        self.evolution = {}
        self.lowest_energy = None
        if loading is False:
            self.directory = self.create_folder(args,info)
        self.displaying = "\n hola, soy VANS :) \n"

    def create_folder(self,args, info):
        if not os.path.exists("TFIM"):
            os.makedirs("TFIM")
        if float(args.noise) > 0:
            if not os.path.exists("TFIM/noisy"):
                os.makedirs("TFIM/noisy")
            name_folder = "TFIM/noisy/"+str(args.n_qubits)+"Q - J "+str(args.J)+" g "+str(args.g)+ " noise "+str(args.noise)
        else:
            name_folder = "TFIM/"+str(args.n_qubits)+"Q - J "+str(args.J)+" g "+str(args.g)
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)
            nrun=0
            final_folder = name_folder+"/run_"+str(nrun)
            with open(name_folder+"/runs.txt", "w+") as f:
                f.write(info)
                f.close()
            os.makedirs(final_folder)
        else:
            folder = os.walk(name_folder)
            nrun=0
            for k in list(folder)[0][1]:
                if k[0]!=".":
                    nrun+=1
            final_folder = name_folder+"/run_"+str(nrun)
            with open(name_folder+"/runs.txt", "r") as f:
                a = f.readlines()[0]
                f.close()
            with open(name_folder+"/runs.txt", "w") as f:
                f.write(str(nrun)+"\n")
                f.write(info)
                f.write("\n")
                f.close()
            os.makedirs(final_folder)
        return final_folder

    def save_dicts_and_displaying(self):
        output = open(self.directory+"/raw_history.pkl", "wb")
        pickle.dump(self.raw_history, output)
        output.close()
        output = open(self.directory+"/evolution.pkl", "wb")
        pickle.dump(self.evolution, output)
        output.close()
        with open(self.directory+"/evolution.txt","w") as f:
            f.write(self.displaying)
            f.close()
        return

    def load_dicts_and_displaying(self,folder):
        opp = open(folder+"/raw_history.pkl" "rb")
        self.raw_history = pickle.load(opp)
        opp = open(folder+"/evolution.pkl", "rb")
        self.evolution = pickle.load(opp)
        with open(folder+"/evolution.txt", "r") as f:
            a = f.readlines()[0]
            f.close()
        self.displaying = f
        return self.displaying

    def accept_energy(self, E, noise=False):
        """
        in the presence of noise, don't give gates for free!

        E: energy after some optimization (to be accepted or not)
        """
        if noise:
            return E < self.lowest_energy
        else:
            return (E-self.lowest_energy)/np.abs(self.lowest_energy) < 0.01

    def add_step(self,indices, resolver, energy, relevant=True):
        """
        indices: list of integers describing circuit to save
        resolver: dictionary with the corresponding circuit's symbols
        energy: expected value of target hamiltonian on prepared circuit.
        relevant: if energy was minimized on that step
        """
        self.raw_history[len(list(self.raw_history.keys()))] = [self.give_unitary(indices, resolver), energy]
        if relevant:
            self.evolution[len(list(self.evolution.keys()))] = [self.give_unitary(indices, resolver), energy]
        if self.lowest_energy is None:
            self.lowest_energy = energy
        return
