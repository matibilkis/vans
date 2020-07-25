def run_circuit(self, list_ops):
    """
    takes as input vector with actions described as integer (given by RL agent)
    """
    wst = VAnsatzSmart(self.n_qubits,
                  self.observable_name,
                  self.target_reward,
                  list_ops)

    if len(wst.symbols) == 0:
        if self.observable_name == "W-state" or self.observable_name is None:
            simulator = cirq.Simulator()
            result = simulator.simulate(wst.get_state(self.qubits, params=np.random.sample(len(wst.symbols))), qubit_order=self.qubits)
            energy = np.trace(np.dot(wst.observable_matrix, cirq.density_matrix_from_state_vector(result.final_state))).real
            probs = np.abs(result.final_state)**2
            return energy, probs
        else:
            ci = tfq.convert_to_tensor([wst.circuit])
            expval = tfq.layers.Expectation()(
                                            ci,
                                            operators=tfq.convert_to_tensor([self.observable]))
            energy = np.squeeze(tf.math.reduce_sum(expval, axis=-1, keepdims=True))
            simulator = cirq.Simulator()
            result = simulator.simulate(wst.get_state(self.qubits), qubit_order=self.qubits)
            probs = np.abs(result.final_state)**2
            return energy, probs

    model = self.vansatz_keras_model(wst, self.observable)
    w_input = tfq.convert_to_tensor([wst.circuit])
    w_output = tf.ones((1, 1))*self.target_reward
    model.fit(x=w_input, y=w_output, batch_size=1, epochs=20, verbose=0)
    energy = float(np.squeeze(model.predict(w_input)))

    simulator = cirq.Simulator()
    result = simulator.simulate(wst.get_state(self.qubits,params=model.get_weights()[0]), qubit_order=self.qubits)
    probs = np.abs(result.final_state)**2
    return energy, probs
