import cirq
import numpy as np
import sympy

class Basic:
    def __init__(self, n_qubits=3):
        """
        n_qubits: number of qubits on your ansatz
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
    def __init__(self, n_qubits=3):
        super(Evaluator, self).__init__(n_qubits=n_qubits)
        self.raw_history = {}
        self.evolution = {}
        self.lowest_energy = None

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
