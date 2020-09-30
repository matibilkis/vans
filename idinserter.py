from circuit_basics import Basic
import numpy as np

class IdInserter(Basic):
    def __init__(self, n_qubits=3,epsilon=0.1, initialization="epsilon"):
        """
        epsilon: perturbation strength
        initialization: how parameters at ientity compilation are perturbated on single-qubit unitary.
                        Options = ["PosNeg", "epsilon"]
        """
        super(IdInserter, self).__init__(n_qubits=n_qubits)
        self.epsilon = epsilon
        self.init_params = initialization


    def randomly_place_almost_identity(self, indexed_circuit, symbol_to_value):
        block_to_insert, insertion_index = self.choose_block(indexed_circuit)
        Iindexed_circuit, Isymbol_to_value, Iindex_to_symbols = self.inserter(indexed_circuit, symbol_to_value, block_to_insert, insertion_index)
        return Iindexed_circuit, Isymbol_to_value, Iindex_to_symbols

    def resolution_2cnots(self, q1, q2):
        """
        sequence of integers describing a CNOT, then unitary (compiled close to identity, rz rx rz) and the same CNOT
        q1: control qubit
        q2: target qubit

        """
        if q1==q2:
            raise Error("SAME QUBIT!")
        rzq1 = self.number_of_cnots + q1
        rzq2 = self.number_of_cnots +  q2
        rxq1 = self.number_of_cnots + self.n_qubits + q1
        rxq2 = self.number_of_cnots + self.n_qubits + q2
        cnot = self.cnots_index[str([q1,q2])]
        return [cnot, rzq1, rxq1, rzq1,  rzq2, rxq2, rzq2, cnot]

    def resolution_1qubit(self, q):
        """
        retrieves rz rx rz on qubit q
        """
        rzq1 = self.number_of_cnots +  q
        rxq1 = self.number_of_cnots + self.n_qubits + q
        return [rzq1, rxq1, rzq1]

    def choose_block(self, indexed_circuit):
        """
        randomly choices an identity resolution and index to place it at indexed_circuit.
        """
        which_block = np.random.choice([0,1], p=[.5,.5])
        insertion_index = np.random.choice(max(1,len(indexed_circuit)))

        if which_block == 0:
            qubit = np.random.choice(self.n_qubits)
            block_to_insert = self.resolution_1qubit(qubit)
        else:
            qubits = np.random.choice(self.n_qubits, 2,replace = False)
            block_to_insert = self.resolution_2cnots(qubits[0], qubits[1])

        return block_to_insert, insertion_index


    def inserter(self, indexed_circuit, symbol_to_value, block_to_insert, insertion_index):
        """
        This funciton loops over the elements of indexed_circuit.

        indexed_circuit: list with integer entries, each one describing a gate.
        block_to_insert: identity resolution to insert right before insertion_index.
        symbol_to_value: parameter resolver for circuit described by indexed_circuit.
        """

        symbols = []
        new_symbols = []
        new_resolver = {}
        full_resolver={} #this is the final output
        full_indices=[] #this is the final output

        par = 0
        for ind, g in enumerate(indexed_circuit):
            if ind == insertion_index:
                if par%3==0:
                    #### PARAMETER INITIALIZATION
                    if self.init_params == "PosNeg":
                        rot = np.random.uniform(-np.pi,np.pi)
                        new_values = [rot, np.random.choice([-1.,1.])*self.epsilon, -rot]
                    else:
                        new_values = [np.random.choice([-1.,1.])*self.epsilon for oo in range(3)]

                for gate in block_to_insert:
                    full_indices.append(gate)
                    if gate < self.number_of_cnots:
                        pass
                    else:
                        qubit = self.qubits[(gate-self.number_of_cnots)%self.n_qubits]
                        new_symbols.append("New_th_"+str(len(new_symbols)))
                        new_resolver[new_symbols[-1]] = new_values[par%3]
                        full_resolver["th_"+str(len(full_resolver.keys()))] = new_resolver[new_symbols[-1]]
                        par+=1

            ### and then go on!
            if 0<= g < self.number_of_cnots:
                full_indices.append(g)

            elif 0<= g-self.number_of_cnots < 2*self.n_qubits:
                full_indices.append(g)
                symbols.append("th_"+str(len(symbols)))
                full_resolver["th_"+str(len(full_resolver.keys()))] = symbol_to_value[symbols[-1]]
            else:
                raise Error("error insertion_block")

        #### now map the index to the (new) symbols, the overhead is run the Basic function again, but it's inexpensive
        _,_, index_to_symbols = self.give_circuit(full_indices)
        symbol_to_value = full_resolver
        return full_indices, symbol_to_value, index_to_symbols
