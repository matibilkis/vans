def optimize(sol, indices, model):

    effective_qubits = list(circuit.all_qubits())

    for k in sol.qubits:#che, lo que no estoy
        if k not in effective_qubits:
            circuit.append(cirq.I.on(k))

    circuit, variables, _ = sol.give_circuit(indices)
    tfqcircuit = tfq.convert_to_tensor([circuit])

    qoutput = tf.ones((1, 1))*sol.lower_bound_Eg
    model.fit(x=tfqcircuit, y=qoutput, batch_size=1, epochs=sol.qepochs, verbose=sol.verbose, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, mode="min")])
    energy = np.squeeze(tf.math.reduce_sum(model.predict(tfqcircuit), axis=-1))
    return energy
