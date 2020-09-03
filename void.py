def prepare_circuit_insertion(self, indexed_circuit, block_to_insert, insertion_index, symbol_to_value,ep=0.01, init_params="PosNeg"):
    """indexed_circuit is a vector with integer entries, each one describing a gate
        block_to_insert is block of unitaries to insert at index insertion
    """
    symbols = []
    new_symbols = []
    new_resolver = {}
    full_resolver={} #this is the final output
    full_indices=[] #this is the final output
    par=0
    #### PARAMETER INITIALIZATION
    #### PARAMETER INITIALIZATION
    if init_params == "PosNeg":
        rot = np.random.uniform(-np.pi,np.pi)
        new_values = [rot, np.random.uniform(-ep,ep), -rot]
    else:
        new_values = [np.random.uniform(-ep,ep) for oo in range(3)]
    #### PARAMETER INITIALIZATION
    #### PARAMETER INITIALIZATION


    for ind, g in enumerate(indexed_circuit):
        #### insert new block ####
        #### insert new block ####

        if ind == insertion_index:
            for gate in block_to_insert:
                full_indices.append(gate)

                if gate < self.number_of_cnots:
                    control, target = self.indexed_cnots[str(gate)]
                else: ### i do only add block of unitaries.
                    qubit = self.qubits[(gate-self.number_of_cnots)%self.n_qubits]


                    #for par, gateblack in zip(range(3),self.parametrized_unitary):
                    new_symbol = "New_th_"+str(len(new_symbols))
                    new_symbols.append(new_symbol)
                    new_resolver[new_symbol] = new_values[par] #rotation around epsilon... we can do it better afterwards
                    full_resolver["th_"+str(len(full_resolver.keys()))] = new_resolver[new_symbol]
                    par+=1
        #### or go on with the rest of the circuit ####
        #### or go on with the rest of the circuit ####
        if 0<= g < self.number_of_cnots:
            full_indices.append(g)
            control, target = self.indexed_cnots[str(g)]

        elif 0<= g-self.number_of_cnots < 2*self.n_qubits:
            new_symbol = "th_"+str(len(symbols))
            symbols.append(new_symbol)
            full_resolver["th_"+str(len(full_resolver.keys()))] = symbol_to_value[new_symbol]
        else:
            print("error insertion_block")

    _,_, index_to_symbols = self.give_circuit(full_indices)

    symbol_to_value = full_resolver

    return full_indices, index_to_symbols, symbol_to_value


def optimize_model_from_indices(self, indexed_circuit, model):
    circuit, variables, _ = self.give_circuit(indexed_circuit)
    effective_qubits = list(circuit.all_qubits())

    for k in self.qubits:#che, lo que no estoy
        if k not in effective_qubits:
            circuit.append(cirq.I.on(k))

    tfqcircuit = tfq.convert_to_tensor([circuit])

    qoutput = tf.ones((1, 1))*self.lower_bound_Eg
    h=model.fit(x=tfqcircuit, y=qoutput, batch_size=1, epochs=self.qepochs,
              verbose=self.verbose, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, mode="min")])
    energy = np.squeeze(tf.math.reduce_sum(model.predict(tfqcircuit), axis=-1))
    return energy,h
