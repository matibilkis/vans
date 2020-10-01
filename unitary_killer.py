from circuit_basics import Basic
import numpy as np
import cirq
import sympy

class UnitaryMurder(Basic):
    def __init__(self, n_qubits=3,  mode="current_gate", g=1, J=0):
        """
        Scans a circuit, evaluates mean value of observable and retrieves a shorter circuit if the energy is not increased too much.
        """
        super(Simplifier, self).__init__(n_qubits=n_qubits)
        self.single_qubit_unitaries = {"rx":cirq.rx, "rz":cirq.rz}
        self.expectation_layer = tfq.layers.Expectation() #this computes hamiltonian's mean value
        self.observable = self.ising_obs(g=g, J=J) #shared with vqe module...

    def ising_obs(self, g=1, J=0):
        self.g=g
        self.J=J
        observable = [-float(0.5*g)*cirq.Z.on(q) for q in self.qubits]
        for q in range(len(self.qubits)):
            observable.append(-float(0.5*J)*cirq.X.on(self.qubits[q])*cirq.X.on(self.qubits[(q+1)%len(self.qubits)]))
        return observable

    def create_proposal_without_gate(self, info_gate):
        """
        Create a circuit without the gate corresponding to info_gate.
        Also, if the new circuit has no gates enough, returns bool value (valid)
        so to not consider this into tfq.expectation_layer.
        """

        index_victim, victim = info_gate

        proposal_circuit=[]
        proposal_symbols_to_values = {}
        prop_cirq_circuit=cirq.Circuit()

        ko=0 #index_of_smymbols_added_to_circuit
        for ind_survivors, gate_survivors in enumerate(indexed_circuit):
            if ind_survivors < self.number_of_cnots:
                proposal_circuit.append(k)
                control, target = self.indexed_cnots[str(k)]
                prop_cirq_circuit.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
            else:
                if ind_survivors != index_victim:
                    proposal_circuit.append(gate_survivors)
                    qubit = self.qubits[(gate_survivors-self.number_of_cnots)%self.n_qubits]
                    new_param = "th_"+str(len(proposal_symbols_to_values.keys()))
                    if 0 <= k-self.number_of_cnots < self.n_qubits:
                        prop_cirq_circuit.append(cirq.rz(sympy.Symbol(new_param)).on(qubit))
                    else:
                        prop_cirq_circuit.append(cirq.rx(sympy.Symbol(new_param)).on(qubit))
                    #### add value to resolver ####
                    proposal_symbols_to_values[new_param] = self.symbol_to_value["th_"+str(ko)]
                ko+=1

        connections, _ = self.scan_qubits(proposal_circuit)
        valid=True
        #now check if we have killed all the gates in a given qubit. If so, will return valid=False
        for q, path in connections.items():
            if len(path) == 0:
                valid = False
            else:
                if ("rx" not in path) and "rz" not in path):
                    valid = False
        return valid, proposal_circuit, proposal_symbols_to_values, prop_cirq_circuit

    def kill_one_unitary(self, indexed_circuit, symbol_to_value, index_to_symbols):
        """
        This method kills one unitary, looping on the circuit and, if finding a parametrized gate, computes the
        energy of a circuit without it.

        If energy is at least %99, then returns the shorter circuit.
        """

        ###### STEP 1: COMPUTE ORIGINAL ENERGY ####
        cirquit = self.give_circuit(indexed_circuit)[0]
        tfq_original_circuit = tfq.convert_to_tensor([cirq.resolve_parameters(cirquit, symbol_to_value)])
        ftq_original_energy = self.expectation_layer( tfq_original_circuit,
                                operators=tfq.convert_to_tensor([self.observable]))
        original_energy = np.float32(np.squeeze(tf.math.reduce_sum(ftq_original_energy, axis=-1, keepdims=True)))

        circuit_proposals=[]
        circuit_proposals_energies=[]

        self.indexed_circuit = indexed_circuit
        self.symbol_to_value = symbol_to_value
        ###### STEP 2: Loop over original circuit. #####
        for index_victim, victim in enumerate(indexed_circuit):
            #this first index will be the one that - potentially - will be deleted
            if j < self.number_of_cnots:
                pass
            else:
                info_gate = [index_victim, victim]
                valid, proposal_circuit, proposal_symbols_to_values, prop_cirq_circuit = self.create_proposal_without_gate(info_gate)
                if valid:

                    tfqcircuit_proposal = tfq.convert_to_tensor([cirq.resolve_parameters(prop_cirq_circuit, proposal_symbols_to_values)])
                    expval_proposal = expectation_layer(  tfqcircuit_proposal,
                                            operators=tfq.convert_to_tensor([self.observable]))
                    proposal_energy = np.float32(np.squeeze(tf.math.reduce_sum(expval_proposal, axis=-1, keepdims=True)))

                    if self.accepting_criteria(proposal_energy, original_energy):
                        circuit_proposals.append([proposal_circuit, proposal_symbols_to_values,proposal_energy])
                        circuit_proposals_energies.append(new_energy)

        ### STEP 3: keep the one of lowest energy (if there're many)
        if len(circuit_proposals)>0:
            favourite = np.where(np.array(circuit_proposals_energies) == np.min(circuit_proposals_energies))[0][0]
            short_circuit, symbol_to_value, energy = circuit_proposals[favourite]
            _,_, index_to_symbols = self.give_circuit(short_circuit)
            return short_circuit, symbol_to_value, index_to_symbols, circuit_proposals_energies[favourite], True
        else:
            return indexed_circuit, symbol_to_value, index_to_symbols, original_energy, False


    def accepting_criteria(self, e_new, e_old):
        """
        we give %1 in exchange of killing an unitary.
        """
        return np.abs(e_new/e_old) > .99


    def scan_qubits(self, indexed_circuit):
        """
        this function scans the circuit as described by {self.indexed_circuit}
        and returns a dictionary with the gates acting on each qubit and the order of appearence on the original circuit.

        It's the same than Simplifier method.
        """
        connections={str(q):[] for q in range(self.n_qubits)} #this saves the gates at each qubit. It does not respects the order.
        places_gates = {str(q):[] for q in range(self.n_qubits)} #this saves, for each gate on each qubit, the position in the original indexed_circuit
        flagged = [False]*len(indexed_circuit) #used to check if you have seen a cnot already, so not to append it twice to the qubit's dictionary

        for nn,idq in enumerate(indexed_circuit): #sweep over all gates in original circuit's list
            for q in range(self.n_qubits): #sweep over all qubits
                if idq<self.number_of_cnots: #if the gate it's a CNOT or not
                    control, target = self.indexed_cnots[str(idq)] #give control and target qubit
                    if q in [control, target] and not flagged[nn]: #if the qubit we are looking at is affected by this CNOT, and we haven't add this CNOT to the dictionary yet
                        connections[str(control)].append(idq)
                        connections[str(target)].append(idq)
                        places_gates[str(control)].append(nn)
                        places_gates[str(target)].append(nn)
                        flagged[nn] = True #so you don't add the other
                else:
                    if (idq-self.number_of_cnots)%self.n_qubits == q: #check if the unitary is applied to the qubit we are looking at
                        if 0 <= idq - self.number_of_cnots< self.n_qubits:
                            connections[str(q)].append("rz")
                        elif self.n_qubits <= idq-self.number_of_cnots <  2*self.n_qubits:
                            connections[str(q)].append("rx")
                        places_gates[str(q)].append(nn)
                    flagged[nn] = True #to check that all gates have been flagged
        ####quick test
        for k in flagged:
            if k is False:
                raise Error("not all flags in flagged are True!")
        return connections, places_gates
