def kill_one_unitary(self, gates_index, resolver, historial):

    circuit_proposals=[] #storing all good candidates.
    circuit_proposals_energies=[]
    for j in gates_index:
        indexed_prop=[]

        prop=cirq.Circuit()
        checking = False
        ko=0
        to_pop=[]

        for k in gates_index:
            if k < self.number_of_cnots:
                indexed_prop.append(k)
                control, target = self.indexed_cnots[str(k)]
                prop.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
            else:
                if k != j:
                    indexed_prop.append(k)
                    qubit = self.qubits[(k-self.number_of_cnots)%self.n_qubits]
                    for par, gate in zip(range(3),self.parametrized_unitary):
                        new_param = "th_"+str(ko)
                        ko+=1
                        prop.append(gate(sympy.Symbol(new_param)).on(qubit))
                else:
                    checking=True
                    for i in range(3):
                        to_pop.append("th_"+str(ko))
                        ko+=1
        if checking is True:
            nr = resolver.copy()
            for p in to_pop:
                nr.pop(p)

            #### check if I actually kill a part of the circuit (for sure this won't help)
            reject=False
            effective_qubits = list(prop.all_qubits())
            for k in self.qubits:
                if k not in effective_qubits:
                    reject=True
                    #prop.append(cirq.I.on(k))
            if reject:
                pass
            else:
                tfqcircuit = tfq.convert_to_tensor([cirq.resolve_parameters(prop, nr)]) ###resolver parameters !!!
                #backend=cirq.DensityMatrixSimulator(noise=cirq.depolarize(self.noise_level))
                expval = tfq.layers.Expectation()(
                                        tfqcircuit,
                                        operators=tfq.convert_to_tensor([self.observable]))
                new_energy = np.float32(np.squeeze(tf.math.reduce_sum(expval, axis=-1, keepdims=True)))
                del expval
                gc.collect()
                if historial.accept_energy(new_energy, kill_one_unitary=True):
                    ordered_resolver = {}
                    for ind,k in enumerate(nr.values()):
                        ordered_resolver["th_"+str(ind)] = k
                    circuit_proposals.append([indexed_prop,ordered_resolver,new_energy])
                    circuit_proposals_energies.append(new_energy)

    if len(circuit_proposals)>0:
        favourite = np.random.choice(len(circuit_proposals))
        short_circuit, resolver, energy = circuit_proposals[favourite]
        simplified=True
        _,_, simplified_index_to_symbols = self.give_circuit(short_circuit)
        return short_circuit, resolver, energy, simplified_index_to_symbols, simplified
    else:
        simplified=False
        return gates_index, resolver, None, None, simplified
